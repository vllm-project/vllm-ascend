#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


def balance_load(requests_by_rank, dev_num, max_num_seqs=None, max_iters=1000, threshold=5):
    modifications = [{"out_blk": [], "in_blk": [], "freeze": False} for _ in range(dev_num)]

    def calc_bubble(cards_info):
        tot_blks = [c["tot_blk"] for c in cards_info]
        max_blk = max(tot_blks)
        avg_blk = sum(tot_blks) / dev_num
        if threshold <= 1:
            return (max_blk - avg_blk) / max_blk if max_blk > 0 else 0.0
        return max_blk - avg_blk

    def perform_swap(run_item, wait_item, card):
        card_idx = card["card_idx"]
        modifications[card_idx]["out_blk"].append(run_item["blk_num"])
        modifications[card_idx]["in_blk"].append(wait_item["blk_num"])
        card["tot_blk"] += wait_item["blk_num"] - run_item["blk_num"]
        card["running"].remove(run_item)
        card["waiting"].remove(wait_item)
        card["running"].append(wait_item)
        card["waiting"].append(run_item)

    def perform_pull_in(wait_item, card):
        card_idx = card["card_idx"]
        modifications[card_idx]["in_blk"].append(wait_item["blk_num"])
        card["tot_blk"] += wait_item["blk_num"]
        card["waiting"].remove(wait_item)
        card["running"].append(wait_item)

    def fill_running_from_waiting(card):
        if max_num_seqs is None:
            return False
        pulled_any = False
        while len(card["running"]) < max_num_seqs and card["waiting"]:
            item = card["waiting"][0]
            item["newly_added"] = True
            perform_pull_in(item, card)
            pulled_any = True
        return pulled_any

    def find_best_swap(card, mode):
        target_avg = sum(c["tot_blk"] for c in cards_data) / dev_num
        best_swap = None
        best_dist = abs(card["tot_blk"] - target_avg)
        best_newly_added = False

        for run_item in card["running"]:
            for wait_item in card["waiting"]:
                delta = wait_item["blk_num"] - run_item["blk_num"]
                if mode == "increase" and delta <= 0:
                    continue
                if mode == "decrease" and delta >= 0:
                    continue
                new_tot = card["tot_blk"] + delta
                new_avg = target_avg + delta / dev_num
                new_dist = abs(new_tot - new_avg)
                run_newly_added = run_item.get("newly_added", False)
                if new_dist < best_dist or (new_dist == best_dist and run_newly_added and not best_newly_added):
                    best_dist = new_dist
                    best_swap = (run_item, wait_item)
                    best_newly_added = run_newly_added
        return best_swap

    def try_drop_for_throughput(max_card):
        card_idx = max_card["card_idx"]
        total_reqs = sum(len(c["running"]) for c in cards_data)
        if total_reqs <= 1:
            return False

        max_blk = max_card["tot_blk"]
        second_max_blk = max([c["tot_blk"] for c in cards_data if c["card_idx"] != max_card["card_idx"]] + [0])
        current_latency = 1200 + 19.2 * max_blk
        current_tp = total_reqs / current_latency
        best_drop = None
        best_tp = current_tp
        best_newly_added = False

        for run_item in max_card["running"]:
            new_max_blk = max(max_blk - run_item["blk_num"], second_max_blk)
            new_latency = 1200 + 19.2 * new_max_blk
            new_tp = (total_reqs - 1) / new_latency
            run_newly_added = run_item.get("newly_added", False)
            if new_tp > best_tp or (new_tp == best_tp and run_newly_added and not best_newly_added):
                best_tp = new_tp
                best_drop = run_item
                best_newly_added = run_newly_added

        if best_drop:
            modifications[card_idx]["out_blk"].append(best_drop["blk_num"])
            max_card["tot_blk"] -= best_drop["blk_num"]
            max_card["running"].remove(best_drop)
            max_card["waiting"].append(best_drop)
            return True
        return False

    cards_data = []
    for rank in range(dev_num):
        blks, split = requests_by_rank[rank] if rank < len(requests_by_rank) else ([], 0)
        running = []
        waiting = []
        tot_run_blk = 0
        for index, blk_num in enumerate(blks):
            if blk_num == 0:
                break
            req_obj = {"blk_num": blk_num}
            if index < split:
                running.append(req_obj)
                tot_run_blk += blk_num
            else:
                waiting.append(req_obj)
        cards_data.append({"card_idx": rank, "running": running, "waiting": waiting, "tot_blk": tot_run_blk})

    for card in cards_data:
        fill_running_from_waiting(card)

    iteration = 0
    while iteration < max_iters:
        cards_data.sort(key=lambda x: x["tot_blk"], reverse=True)
        if calc_bubble(cards_data) < threshold:
            break

        swapped_this_round = False
        dropped_this_round = False
        min_card = cards_data[-1]
        best_min_swap = find_best_swap(min_card, mode="increase")
        if best_min_swap:
            perform_swap(best_min_swap[0], best_min_swap[1], min_card)
            swapped_this_round = True

        if swapped_this_round:
            if calc_bubble(cards_data) < threshold:
                break
            cards_data.sort(key=lambda x: x["tot_blk"], reverse=True)

        max_card = cards_data[0]
        best_max_swap = find_best_swap(max_card, mode="decrease")
        if best_max_swap:
            perform_swap(best_max_swap[0], best_max_swap[1], max_card)
            swapped_this_round = True

        if not swapped_this_round and try_drop_for_throughput(max_card):
            dropped_this_round = True

        if calc_bubble(cards_data) < threshold:
            break
        if not swapped_this_round and not dropped_this_round:
            break
        iteration += 1

    for mod in modifications:
        out_blks = mod["out_blk"][:]
        in_blks = mod["in_blk"][:]
        cancelled_out = [False] * len(out_blks)
        cancelled_in = [False] * len(in_blks)
        for in_index, in_blk in enumerate(in_blks):
            for out_index, out_blk in enumerate(out_blks):
                if not cancelled_out[out_index] and out_blk == in_blk:
                    cancelled_out[out_index] = True
                    cancelled_in[in_index] = True
                    break
        mod["out_blk"] = [b for b, cancelled in zip(mod["out_blk"], cancelled_out) if not cancelled]
        mod["in_blk"] = [b for b, cancelled in zip(mod["in_blk"], cancelled_in) if not cancelled]

    for card in cards_data:
        rank = card["card_idx"]
        modifications[rank]["freeze"] = len(modifications[rank]["in_blk"]) == 0

    return modifications
