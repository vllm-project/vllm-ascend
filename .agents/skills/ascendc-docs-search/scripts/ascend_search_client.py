"""Ascend Search Client 华为昇腾社区文档搜索客户端"""
import sys
from ascend_search_skill import AscendSearchSkill


def search_ascend_docs(keyword, lang="zh", max_results=10, doc_type="DOC", page_num=1, version_filter=None):
    """
    搜索华为昇腾社区文档并返回解析后的内容

    Args:
        keyword: 搜索关键词
        lang: 语言，默认中文
        max_results: 最大返回结果数量
        doc_type: 文档类型，DOC（文档）或 API（API文档）
        page_num: 页码
        version_filter: 版本过滤字符串，如果提供则只返回包含该字符串的版本

    Returns:
        解析后的结构化搜索结果
    """
    try:
        # 创建搜索技能实例
        skill = AscendSearchSkill()
        
        # 执行搜索
        print(f"🔍 正在搜索: {keyword}")
        result = skill.search_documents(
            keyword=keyword,
            lang=lang,
            doc_type=doc_type,
            page_num=page_num,
            page_size=max_results
        )
        
        # 检查搜索是否成功
        if not result.get("success"):
            error_msg = result.get("message", "搜索失败")
            print(f"❌ 搜索失败: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "results": []
            }
        
        # 解析和格式化结果
        data = result.get("data", [])
        parsed_results = []
        
        for item in data:
            parsed_item = {
                "title": item.get("title", "无标题"),
                "summary": item.get("summary", "无摘要"),
                "url": item.get("url", ""),
                "version": item.get("version", "未知"),
                "content_type": item.get("content_type", "文档"),
                "relevance_score": item.get("relevance_score", 0)
            }
            
            # 生成简短摘要用于显示
            summary = parsed_item["summary"]
            if summary and len(summary) > 120:
                parsed_item["display_summary"] = summary[:120] + "..."
            else:
                parsed_item["display_summary"] = summary
                
            parsed_results.append(parsed_item)

        # 版本过滤
        filtered_results = parsed_results
        if version_filter:
            original_count = len(parsed_results)
            filtered_results = [
                item for item in parsed_results
                if version_filter.lower() in item.get("version", "").lower()
            ]
            filtered_count = len(filtered_results)
            print(f"🔍 版本过滤: '{version_filter}'")
            print(f"📊 过滤前: {original_count} 个文档 | 过滤后: {filtered_count} 个文档")
            if filtered_count == 0:
                print("⚠️  警告: 没有找到匹配指定版本的文档")

        # 显示搜索结果
        print(f"📊 找到 {len(filtered_results)} 个相关文档")
        print("=" * 70)
        
        for i, item in enumerate(filtered_results, 1):
            print(f"\n{i}. {item['title']}")
            print(f"   摘要: {item['display_summary']}")
            print(f"   链接: {item['url']}")
            print(f"   版本: {item['version']} | 类型: {item['content_type']}")
            if item.get('relevance_score'):
                print(f"   相关度: {item['relevance_score']}")
            print("-" * 50)
        
        # 返回结构化结果
        return {
            "success": True,
            "keyword": keyword,
            "version_filter": version_filter,
            "original_count": len(parsed_results),
            "filtered_count": len(filtered_results),
            "results": filtered_results
        }
        
    except Exception as e:
        error_msg = f"搜索过程中发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "results": []
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="华为昇腾社区文档搜索客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python ascend_search_client.py "Ascend C API"
  python ascend_search_client.py "模型训练教程" --lang zh --max_results 5
  python ascend_search_client.py "算子开发" --lang en --doc_type API --page_num 1
  python ascend_search_client.py "Ascend C" --version "8.3.RC1"
  python ascend_search_client.py "算子开发" --version "社区版" --max_results 8
        """
    )

    # 位置参数：搜索关键词
    parser.add_argument(
        "keyword",
        type=str,
        help="搜索关键词，建议使用中文"
    )

    # 可选参数
    parser.add_argument(
        "--lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="语言设置，默认为中文 (zh)"
    )

    parser.add_argument(
        "--max_results",
        type=int,
        default=10,
        choices=range(1, 11),
        metavar="[1-10]",
        help="返回的最大结果数量，范围1-10，默认为10"
    )

    parser.add_argument(
        "--page_size",
        type=int,
        dest="max_results",
        help="与 --max_results 相同，用于兼容性"
    )

    parser.add_argument(
        "--doc_type",
        type=str,
        default="DOC",
        choices=["DOC", "API"],
        help="文档类型：DOC（文档）或 API（API文档），默认为 DOC"
    )

    parser.add_argument(
        "--page_num",
        type=int,
        default=1,
        choices=range(1, 101),
        metavar="[1-100]",
        help="页码，范围1-100，默认为1"
    )

    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="版本过滤字符串，只返回版本信息中包含该字符串的文档（不区分大小写）"
    )

    args = parser.parse_args()

    # 如果同时指定了 --max_results 和 --page_size，优先使用 --max_results
    # argparse 已经处理了 dest 映射，这里只需要检查是否存在冲突
    # 实际参数已存储在 args.max_results 中

    # 执行搜索
    search_ascend_docs(
        keyword=args.keyword,
        lang=args.lang,
        max_results=args.max_results,
        doc_type=args.doc_type,
        page_num=args.page_num,
        version_filter=args.version
    )