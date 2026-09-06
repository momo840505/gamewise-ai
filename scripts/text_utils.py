from __future__ import annotations

import re

import pandas as pd


TRADITIONAL_CHINESE_QUERY_SYNONYMS: dict[str, str] = {
    "免費遊戲": " free game ",
    "免費": " free ",
    "免费游戏": " free game ",
    "免费": " free ",
    "不用錢": " free ",
    "不用钱": " free ",
    "零元": " free ",
    "預算": " budget ",
    "预算": " budget ",
    "價錢": " price ",
    "价钱": " price ",
    "價格": " price ",
    "价格": " price ",
    "售價": " price ",
    "售价": " price ",
    "花費": " price ",
    "花费": " price ",
    "以下": " under ",
    "以內": " under ",
    "以内": " under ",
    "之內": " under ",
    "之内": " under ",
    "低於": " under ",
    "低于": " under ",
    "不超過": " under ",
    "不超过": " under ",
    "不能超過": " under ",
    "不能超过": " under ",
    "小於": " under ",
    "小于": " under ",
    "少於": " under ",
    "少于": " under ",
    "不到": " under ",
    "便宜": " cheap ",
    "美元": " ",
    "美金": " ",
    "鎂": " ",
    "元以下": " under ",
    "元以內": " under ",
    "元以内": " under ",
    "塊以下": " under ",
    "块以下": " under ",
    "好評": " positive reviews ",
    "好评": " positive reviews ",
    "正面評價": " positive reviews ",
    "正面评价": " positive reviews ",
    "正評": " positive reviews ",
    "正评": " positive reviews ",
    "正面": " positive reviews ",
    "評價": " rating ",
    "评价": " rating ",
    "評分": " rating ",
    "評價率": " rating ",
    "评价率": " rating ",
    "推薦率": " positive reviews ",
    "推荐率": " positive reviews ",
    "玩家評價": " positive reviews ",
    "玩家评价": " positive reviews ",
    "至少": " at least ",
    "起碼": " at least ",
    "起码": " at least ",
    "至少要": " at least ",
    "以上": " at least ",
    "高於": " over ",
    "高于": " over ",
    "超過": " over ",
    "超过": " over ",
    "不低於": " at least ",
    "不低于": " at least ",
    "支援": " for ",
    "支持": " for ",
    "適合": " for ",
    "适合": " for ",
    "可玩": " for ",
    "可以玩": " for ",
    "多人合作": " co-op ",
    "多人协作": " co-op ",
    "連線合作": " online co-op ",
    "联机合作": " online co-op ",
    "線上合作": " online co-op ",
    "线上合作": " online co-op ",
    "本地合作": " local co-op ",
    "可以合作": " co-op ",
    "可合作": " co-op ",
    "合作遊玩": " co-op ",
    "合作游玩": " co-op ",
    "一起玩": " co-op ",
    "一起打": " co-op ",
    "朋友一起": " co-op ",
    "雙人": " co-op ",
    "双人": " co-op ",
    "兩人": " co-op ",
    "两人": " co-op ",
    "單人": " single-player ",
    "单人": " single-player ",
    "單機": " single-player ",
    "单机": " single-player ",
    "一個人": " single-player ",
    "一个人": " single-player ",
    "自己玩": " single-player ",
    "獨自": " single-player ",
    "独自": " single-player ",
    "多人連線": " multiplayer ",
    "多人联机": " multiplayer ",
    "多人連機": " multiplayer ",
    "連機": " multiplayer ",
    "联机": " multiplayer ",
    "線上多人": " multiplayer ",
    "线上多人": " multiplayer ",
    "網路多人": " multiplayer ",
    "网络多人": " multiplayer ",
    "多人遊戲": " multiplayer ",
    "多人游戏": " multiplayer ",
    "多人": " multiplayer ",
    "生存遊戲": " survival game ",
    "生存游戏": " survival game ",
    "生存": " survival ",
    "求生": " survival ",
    "活下去": " survival ",
    "末日": " survival ",
    "喪屍": " survival ",
    "丧尸": " survival ",
    "殭屍": " survival ",
    "僵尸": " survival ",
    "心理恐怖": " psychological horror ",
    "精神恐怖": " psychological horror ",
    "恐怖": " horror ",
    "驚悚": " horror ",
    "惊悚": " horror ",
    "嚇人": " scary horror ",
    "吓人": " scary horror ",
    "詭異": " creepy horror ",
    "诡异": " creepy horror ",
    "放鬆": " relaxing ",
    "放松": " relaxing ",
    "輕鬆": " relaxing ",
    "轻松": " relaxing ",
    "療癒": " cozy ",
    "疗愈": " cozy ",
    "治癒": " cozy ",
    "治愈": " cozy ",
    "溫馨": " cozy ",
    "温馨": " cozy ",
    "休閒": " casual ",
    "休闲": " casual ",
    "簡單": " casual ",
    "简单": " casual ",
    "輕度": " casual ",
    "轻度": " casual ",
    "回合制": " turn-based ",
    "回合": " turn-based ",
    "回合策略": " turn-based strategy ",
    "戰術": " tactical ",
    "战术": " tactical ",
    "戰棋": " tactical ",
    "战棋": " tactical ",
    "策略": " strategy ",
    "戰略": " strategy ",
    "战略": " strategy ",
    "即時戰略": " real time strategy ",
    "即时战略": " real time strategy ",
    "4X": " 4x ",
    "冒險": " adventure ",
    "冒险": " adventure ",
    "探索": " exploration ",
    "解謎": " puzzle ",
    "解谜": " puzzle ",
    "謎題": " puzzle ",
    "谜题": " puzzle ",
    "推理": " puzzle ",
    "農場": " farming ",
    "农场": " farming ",
    "種田": " farming ",
    "种田": " farming ",
    "耕種": " farming ",
    "耕种": " farming ",
    "牧場": " farming ",
    "牧场": " farming ",
    "模擬": " simulation ",
    "模拟": " simulation ",
    "經營": " simulation ",
    "经营": " simulation ",
    "管理": " management ",
    "養成": " simulation ",
    "养成": " simulation ",
    "劇情": " story rich ",
    "剧情": " story rich ",
    "故事": " story rich ",
    "敘事": " story rich ",
    "叙事": " story rich ",
    "多結局": " multiple endings ",
    "多结局": " multiple endings ",
    "角色扮演": " rpg ",
    "日式RPG": " jrpg ",
    "日式rpg": " jrpg ",
    "動作RPG": " action rpg ",
    "動作rpg": " action rpg ",
    "动作RPG": " action rpg ",
    "动作rpg": " action rpg ",
    "賽車": " racing ",
    "赛车": " racing ",
    "競速": " racing ",
    "竞速": " racing ",
    "開車": " driving ",
    "开车": " driving ",
    "射擊": " shooter ",
    "射击": " shooter ",
    "槍戰": " shooter ",
    "枪战": " shooter ",
    "第一人稱射擊": " first person shooter ",
    "第一人称射击": " first person shooter ",
    "第三人稱射擊": " third person shooter ",
    "第三人称射击": " third person shooter ",
    "開放世界": " open world ",
    "开放世界": " open world ",
    "開放式世界": " open world ",
    "开放式世界": " open world ",
    "沙盒": " sandbox ",
    "年之後": " after ",
    "年之后": " after ",
    "年後": " after ",
    "年后": " after ",
    "年以前": " before ",
    "年之前": " before ",
    "之後": " after ",
    "之后": " after ",
    "以後": " after ",
    "以后": " after ",
    "以前": " before ",
    "之前": " before ",
}


def expand_traditional_chinese_query(query: str) -> str:
    """Append English equivalents for common Traditional Chinese game queries."""

    # BUG FIX: this used to be a single plain `term in query` containment
    # check for every dictionary entry. That is the same substring-match
    # problem the extract_filters comments above describe, and it bites
    # here first: "多人" ("multiplayer") is also the tail of "很多人" /
    # "好多人" / "許多人" ("a lot of people"), e.g. "很多人推薦這款遊戲"
    # ("a lot of people recommend this game"). A plain substring check
    # silently appended the English word "multiplayer" to the expanded
    # query, which extract_filters then picked up as a real play-mode
    # request. "多人" is special-cased with a negative lookbehind for the
    # known "many people" prefixes instead of a bare containment check.
    #
    # The dictionary also used to have bare entries for "合作"/"協作"/
    # "協力" ("cooperate"/"collaborate") and "跟朋友"/"和朋友"/"與朋友"
    # ("with a friend"), which are ordinary Chinese words used constantly
    # outside any game-mode context, e.g. "和知名動畫合作推出" ("released
    # in collaboration with a well-known anime") or "跟朋友討論過這款遊戲"
    # ("discussed this game with a friend"). Both silently forced
    # play_mode to co-op. Rather than chase every non-gameplay use of
    # those words with more lookarounds, they were removed in favor of
    # the more specific phrases already in this dictionary that actually
    # mean "play together" (可以合作, 合作遊玩, 一起玩, 朋友一起, etc.).
    # Same reasoning for "選擇" ("choose"), which mapped to "choices
    # matter" but is mostly just the verb "to choose" in a request like
    # "幫我選擇一款遊戲" ("help me choose a game") -- "多結局" ("multiple
    # endings") already covers that concept without the ambiguity.
    expanded_terms = []
    for term, replacement in TRADITIONAL_CHINESE_QUERY_SYNONYMS.items():
        if term == "多人":
            if re.search(r"(?<![很好許许眾众])多人", query):
                expanded_terms.append(replacement)
            continue
        if term in query:
            expanded_terms.append(replacement)

    if not expanded_terms:
        return query

    return query + " " + " ".join(expanded_terms)


def get_optional_text_column(
    dataframe: pd.DataFrame,
    column_name: str,
) -> pd.Series:
    """
    Return a column or an aligned empty column.

    This lets tests use a smaller DataFrame without
    production-only metadata such as categories.
    """

    if column_name in dataframe.columns:
        return dataframe[
            column_name
        ]

    return pd.Series(
        "",
        index=dataframe.index,
        dtype="object",
    )


def category_contains(
    category_series: pd.Series,
    pattern: str,
) -> pd.Series:
    """Search official Steam category text."""

    return (
        category_series
        .fillna("")
        .astype(str)
        .str.contains(
            pattern,
            case=False,
            regex=True,
            na=False,
        )
    )
