import math
from typing import Optional

import torch
from vllm.config import VllmConfig
from vllm.transformers_utils.configs.hstu_config import \
    HSTUInferenceRankingConfig


@torch.no_grad()
def truncated_normal_(tensor, mean=0.0, std=0.02, lower=-2.0, upper=2.0):
    """
    Fills the input tensor with values drawn from a truncated normal distribution.

    Args:
        tensor (torch.Tensor): an n-dimensional tensor
        mean (float): mean of the normal distribution
        std (float): standard deviation of the normal distribution
        lower (float): lower bound (in terms of number of standard deviations)
        upper (float): upper bound (in terms of number of standard deviations)

    Returns:
        None. Fills the input tensor in-place.
    """
    size = tensor.size()
    tmp = tensor.new_empty(size).normal_()

    tmp = tmp.clamp(min=lower, max=upper)

    tensor.copy_(tmp)
    tensor.mul_(std).add_(mean)


class InferenceEmbedding(torch.nn.Module):
    """
    InferenceEmbedding is a module for embeddings in the inference stage.

    Args:
        embedding_configs (List[InferenceEmbeddingConfig]): Configuration for the hstu (sharded) embedding.
        embedding_dim (Optional[int]): Dim of hstu embedding.
        is_merged_table (Optional[bool]): Whether to merge embedding table.
        table_type_seq (Optional[List[str]]): The seq of different embedding tables.
    """

    def __init__(
        self,
        hf_config: HSTUInferenceRankingConfig,
        max_num_batched_tokens: int = 0,
    ):
        super().__init__()

        embedding_configs = hf_config.task_config.embedding_configs
        self._is_merged_table = hf_config.merged_table if hf_config.merged_table is not None else True
        self._embedding_dim = hf_config.hidden_size
        self._dropout_rate = hf_config.hstu_config.dropout_ratio
        self._feature_sum_dim = 0
        if self._is_merged_table:
            self._sum_vocab_size = 0
            self._feature_cnt = 0
            embedding_table_name = []
            for config in embedding_configs:
                if config.table_name not in embedding_table_name:
                    self._sum_vocab_size += config.vocab_size
                    self._feature_cnt += 1
                    dim = config.dim
                    embedding_table_name.append(config.table_name)
                if config.table_name == 'item':
                    self._feature_sum_dim += config.dim
            self._embedding_layer = torch.nn.Embedding(
                num_embeddings=self._sum_vocab_size,
                embedding_dim=dim,
            )

            # # 直接创建parameter降低显存占用
            # self.weight = torch.nn.Parameter(
            #     torch.randn(self._sum_vocab_size, dim, dtype=torch.bfloat16, device='npu'),
            #     requires_grad=False
            # )

            # 多个输入特征embedding拼接后过一个mlp
            # TODO 单个输入特征也增加一个mlp
            # if self._feature_cnt != 2 or True:
            self._emb_mlp = torch.nn.Linear(self._feature_sum_dim,
                                            self._embedding_dim)

            self._pos_emb = torch.nn.Embedding(
                num_embeddings=max_num_batched_tokens,
                embedding_dim=dim,
            )
            self._emb_dropout = torch.nn.Dropout(p=self._dropout_rate)
        else:
            self._features_seq = hf_config.input_seq
            self._embs_feature_cnt = hf_config.task_config.embedding_feature_cnt
            self._table_type = []
            self._multi_value_prefix = hf_config.multi_value_prefix
            self._embedding_layer = torch.nn.ModuleDict()
            table_2_embedding_dim = {}
            for config in embedding_configs:
                feature_name = config.feature_names[0]
                table_type = config.table_name
                vocab_size = config.vocab_size
                dim = config.dim
                associated_feature_table = config.associated_feature_table
                associated_feature_name = config.associated_feature_name
                if table_type not in table_2_embedding_dim:
                    table_2_embedding_dim[table_type] = 0
                    self._table_type.append(table_type)
                table_2_embedding_dim[table_type] += dim
                if associated_feature_table and associated_feature_name:
                    self._embedding_layer[
                        feature_name] = self._embedding_layer[
                            associated_feature_name]
                else:
                    self._embedding_layer[feature_name] = torch.nn.Embedding(
                        num_embeddings=vocab_size,
                        embedding_dim=dim,
                    )
            self._embs_mlp = torch.nn.ModuleDict()
            for table_type, _ in table_2_embedding_dim.items():
                if self._embedding_dim != 0:
                    self._embs_mlp[table_type] = torch.nn.Linear(
                        table_2_embedding_dim[table_type], self._embedding_dim)

    def to_empty(self, device: torch.device):
        super().to_empty(device=device)

        @torch.no_grad()
        def init_embedding_weights(m):
            if isinstance(m, torch.nn.Embedding):
                truncated_normal_(m.weight, mean=0.0, std=0.02)

        self.apply(init_embedding_weights)
        torch.nn.init.normal_(self._emb_mlp.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        is_prefill: Optional[bool] = True,
    ) -> torch.Tensor:
        if self._is_merged_table:

            def process_embs(x, mlp, feature_cnt):
                B, D = x.shape
                assert B % feature_cnt == 0, f"T must be divisible by {feature_cnt}, got {B}"

                N = B // feature_cnt
                x = x.view(N, feature_cnt, D)

                features = x[:, :feature_cnt - 1, :]  # (N, feature_cnt - 1, D)
                action = x[:, feature_cnt - 1:, :]  # (N, 1, D)

                features = features.view(N, (feature_cnt - 1) *
                                         D)  # (N, (feature_cnt - 1) * D))
                features = mlp(features)  # (N, D)
                features = features.unsqueeze(-2)  # (N, 1, D)

                result = torch.stack([features, action],
                                     dim=1).reshape(action.shape[0] * 2, D)

                return result

            embs = self._embedding_layer(input_ids)
            if hasattr(self, '_emb_mlp'):
                embs = process_embs(embs, self._emb_mlp, self._feature_cnt)
            embs = embs * (self._embedding_dim**0.5)
            embs = embs + self._pos_emb(position_ids)
            embs = self._emb_dropout(embs)

            if is_prefill:
                # 取出action
                action = input_ids[1::2]
                mask = (action != 0).unsqueeze(1).expand(-1, 2).reshape(-1, 1)
                embs = embs * mask
            return embs
        else:
            embeddings = []
            embedding_res = []
            table_index = 0
            feature_type_index = 0
            feature_index = 0
            sum_cnt = 0
            for feature_name in self._features_seq:
                if "pref_" in feature_name:
                    embedding_pref = self._embedding_layer[feature_name](
                        input_ids[feature_type_index][feature_index])
                    feature_dim = embedding_pref.size(-1)
                    feat_mask = input_ids[feature_type_index][
                        feature_index] != 0
                    feat_mask = feat_mask.unsqueeze(-1).repeat(
                        1, 1, feature_dim)
                    # 对pref多值特征做mean pooling，并忽略掉值为self.padding_index的填充index
                    embedding_pref = (embedding_pref * feat_mask).sum(
                        dim=1) / (feat_mask.sum(dim=1) + 1e-6)
                    embeddings.append(embedding_pref)
                else:
                    embeddings.append(self._embedding_layer[feature_name](
                        input_ids[feature_type_index][feature_index]))
                feature_index += 1
                sum_cnt += 1
                if feature_index >= input_ids[feature_type_index].shape[0]:
                    feature_type_index += 1
                    feature_index = 0

                if sum_cnt == sum(self._embs_feature_cnt[:table_index + 1]):
                    embeddings_i = torch.cat(embeddings, dim=-1)
                    embeddings_i = self._embs_mlp[
                        self._table_type[table_index]](embeddings_i)
                    embedding_res.append(embeddings_i)
                    table_index += 1
                    embeddings = []

            return embedding_res


class InputPreprocessModule(torch.nn.Module):

    def __init__(self, config: VllmConfig):
        super().__init__()

        hf_config = config.model_config.hf_config
        hstu_config = hf_config.hstu_config
        max_sequence_len = hstu_config.max_preprocess_len
        # gr_output_length = hf_config.gr_output_length
        # max_sequence_len = max_sequence_length + gr_output_length
        self._embedding_dim: int = hstu_config.hidden_size
        num_ratings = hf_config.num_ratings if hf_config.num_ratings else 5
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len * 2 + 2,
            self._embedding_dim,
        )
        self._dropout_rate: float = hstu_config.dropout_ratio
        self._emb_dropout = torch.nn.Dropout(p=self._dropout_rate)
        self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_ratings + 1, self._embedding_dim, padding_idx=0)
        self.num_ratings = num_ratings
        # self._infer_ratings_key = hf_config.get("infer_ratings_key", "ratings")
        self.reset_state()

    def reset_state(self) -> None:

        def weird_division(x, y):
            """
            :param x: 被除数
            :param y: 除数
            :return x/y: 商
            """
            EPS = 1e-7
            if y < EPS and y >= 0:
                y = EPS
            elif y > -EPS and y < 0:
                y = -EPS
            return x / y

        truncated_normal_(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(weird_division(1.0, self._embedding_dim)),
        )
        truncated_normal_(
            self._rating_emb.weight.data,
            mean=0.0,
            std=math.sqrt(weird_division(1.0, self._embedding_dim)),
        )

    def get_preprocessed_masks(
        self,
        past_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        生成预处理后的掩码。
        
        :param past_ids: 历史ID张量。
        :return: 预处理后的掩码张量, 形状为(B, N * 2)。
        """
        B, N = past_ids.size()
        return (past_ids != 0).unsqueeze(2).expand(-1, -1, 2).reshape(B, N * 2)

    def forward(
        self,
        # past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        user_feature_embs: torch.Tensor,
        past_item_embeddings: torch.Tensor,
        past_ratings: torch.Tensor,
        num_rerank: int = 0,
    ):
        '''
        前向传播方法, 处理用户、物品和评分的特征

        :param past_lengths: 历史长度张量, 形状为(B,), 其中B是批次大小, 表示每个序列的长度。
        :param past_ids: 历史ID张量, 形状为(B, N), 其中N是序列中的最大项数, 表示每个序列中的物品ID。
        :param user_feature_embs: 用户特征embedding张量, 形状为(B, D), D是用户特征embedding的维度。
        :param past_item_embeddings: 历史embedding张量, 形状为(B, N, D), 包含序列中每个物品的embedding表示。
        :param past_ratings: 历史物品评分。
        :return: 新的序列长度, 形状为(B,), 是原始有效序列长度的两倍+1。
                 预处理后的用户特征embedding, 形状为(B, 1+N*2+1, D), 将用户特征、物品特征和评分特征结合起来最后补0。
                 有效掩码张量, 形状为(B, N*2), 用于指示哪些位置是有效的, 即非零ID的位置。  
        '''
        past_ids = past_ids.unsqueeze(0)
        past_item_embeddings = past_item_embeddings.unsqueeze(0)
        past_ratings = past_ratings.unsqueeze(0)

        B, N = past_ids.size()
        D = past_item_embeddings.size(-1)

        # 提取评分 embedding
        rating_emb = self._rating_emb(past_ratings)

        # 拼接历史物品 embedding 与评分 embedding, (i1,i2,i3,...), (a1,a2,a3,...)->(i1,a1,i2,a2,i3,a3,...)
        user_embeddings = torch.cat(
            [past_item_embeddings, rating_emb],
            dim=2,
        ) * (self._embedding_dim**0.5)
        user_embeddings = user_embeddings.view(B, N * 2, D)
        user_embeddings = (user_embeddings + self._pos_emb(
            torch.arange(N * 2, device=past_ids.device).unsqueeze(0).repeat(
                B, 1)))
        user_embeddings = self._emb_dropout(user_embeddings)

        # 生成有效掩码并应用
        valid_mask = self.get_preprocessed_masks(past_ids).unsqueeze(2).float()
        user_embeddings *= valid_mask
        user_embeddings = torch.cat(
            (user_feature_embs.unsqueeze(1), user_embeddings), dim=1)
        # 为了适用于底层加速，需要使user_embeddings序列长度为偶数
        if num_rerank == 0:
            user_embeddings = torch.nn.functional.pad(user_embeddings,
                                                      (0, 0, 0, 1, 0, 0),
                                                      'constant', 0.0)

        return user_embeddings.squeeze(0), valid_mask

    def process_rerank_embs(
            self,
            rerank_embs: torch.Tensor,  # B, NUM_CANDIDATE, D
            past_lengths: torch.Tensor,  # B, 1
    ):
        rerank_embs = rerank_embs.unsqueeze(0)
        B, N, D = rerank_embs.shape
        position_embs = self._pos_emb(past_lengths - 1).unsqueeze(1).repeat(
            1, N, 1)  # B x NUM_CANDIDATE x D
        rerank_embs = rerank_embs * (self._embedding_dim**0.5) + position_embs
        rerank_embs = torch.nn.functional.pad(rerank_embs, (0, 0, 0, 1, 0, 0),
                                              'constant', 0.0)
        rerank_embs.squeeze(0)
        return rerank_embs
