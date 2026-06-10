# -*- coding: utf-8 -*-
"""
==============================================================================
 硬科技【紫苏叶 AI 投研与综合择时系统】
==============================================================================
 交易哲学：
   1. 紫苏叶选股：寻找产业链深层节点（Layer3+）、不可替代、寡头垄断（竞争对手<=3家）的底层硬件/材料公司。
   2. 戴维斯双击：利润暴增（净利润同比增长率 > 20%）且估值偏低（PE 处于近3年50%分位以下）。
   3. 右侧交易+筹码支撑：均线多头排列且筹码低位单峰密集时买入，破位则坚决卖出。

 模块结构：
   模块0  基础设施：SQLite 建表、网络容错封装、缓存
   模块1  紫苏叶守门员（DeepSeek 基本面研判）
   模块2  动态数据拉取（akshare 行情/财务/龙虎榜）
   模块3  AI 视觉分析师（Gemini 筹码分布图分析）
   模块4  终极买卖决策引擎
   模块5  Streamlit 小白友好 UI

 设计原则：界面通俗易懂；专业术语配大白话解释；结论给"该怎么做 + 为什么"。
==============================================================================
"""

import os
import re
import hmac
import json
import sqlite3
import datetime
from io import BytesIO

import pandas as pd
import requests
import streamlit as st

# 视觉与图片库在使用时再导入，避免环境缺失时整页崩溃
try:
    import google.generativeai as genai
    _GENAI_OK = True
except Exception:
    _GENAI_OK = False

try:
    from PIL import Image
    _PIL_OK = True
except Exception:
    _PIL_OK = False


# ============================================================================
# 模块0：基础设施（数据库、常量、网络容错）
# ============================================================================

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perilla_stock.db")  # 默认（未登录）数据库文件
MISSING = "数据缺失"          # 取数失败时的统一占位


def _safe_user_key(username: str) -> str:
    """把用户名转成安全的文件名片段（只保留字母/数字/下划线，其余替换为下划线）。"""
    safe = re.sub(r"[^0-9A-Za-z_]", "_", (username or "").strip())
    return safe or "user"


def current_db_path():
    """
    返回"当前登录用户专属"的数据库文件路径，实现『每个人各看各的股票池』。
    未登录时回退到默认库。每个用户一个独立的 .db 文件，互不可见。
    """
    user = st.session_state.get("auth_user")
    if user:
        base = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base, f"perilla_stock_{_safe_user_key(user)}.db")
    return DB_PATH
HTTP_TIMEOUT = 30             # 一般网络请求超时（秒）
# DeepSeek 大模型回答较慢，用 (连接超时, 读取超时)：连接 15 秒、读取 180 秒
DEEPSEEK_TIMEOUT = (15, 180)
DEEPSEEK_RETRIES = 2          # 超时/网络抖动时自动重试次数

# 操作指令的展示样式：emoji + 中文标签 + 卡片背景色
SIGNAL_STYLE = {
    "强烈买入":   {"emoji": "🌟", "color": "#ffd60a", "text": "#5a4b00"},
    "分批建仓买入": {"emoji": "🟢", "color": "#2ecc71", "text": "#ffffff"},
    "持有移动止盈": {"emoji": "🔵", "color": "#3498db", "text": "#ffffff"},
    "只看不动观望": {"emoji": "🟡", "color": "#f1c40f", "text": "#5a4b00"},
    "坚决清仓卖出": {"emoji": "🔴", "color": "#e74c3c", "text": "#ffffff"},
    "待补充筹码数据": {"emoji": "🟤", "color": "#95a5a6", "text": "#ffffff"},
    "待刷新行情数据": {"emoji": "🟠", "color": "#e67e22", "text": "#ffffff"},
}


def init_db():
    """初始化数据库，创建股票池表（若不存在）。库文件按登录用户隔离。"""
    conn = sqlite3.connect(current_db_path())
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_pool (
            code             TEXT PRIMARY KEY,   -- 股票代码（6位）
            name             TEXT,               -- 股票简称
            is_perilla_leaf  INTEGER DEFAULT 1,  -- 是否紫苏叶（1是/0否）
            analysis         TEXT,               -- 守门员分析理由
            is_holding       INTEGER DEFAULT 0,  -- 是否已持仓（1是/0否）
            npr_growth       REAL,               -- 净利润同比增长率（%）
            eps              REAL,               -- 基本每股收益 EPS（同花顺源，最新报告期）
            pe               REAL,               -- 当前市盈率 PE
            pe_percentile    REAL,               -- PE 处于近3年的分位（%）
            close            REAL,               -- 最新收盘价
            ma10             REAL,               -- 10日均价线
            ma20             REAL,               -- 20日均价线
            ma30             REAL,               -- 30日均价线
            lhb_flag         INTEGER DEFAULT 0,  -- 近一交易日是否上龙虎榜
            chip_single_peak INTEGER,            -- 筹码：低位单峰密集（1是/0否）
            chip_above_avg   INTEGER,            -- 筹码：股价站上平均成本线
            chip_high_diverge INTEGER,           -- 筹码：高位发散（1是/0否）
            chip_confidence  REAL,               -- 视觉模型置信度（0-1）
            chip_manual      INTEGER DEFAULT 0,  -- 筹码结论是否被人工修正过
            has_chip         INTEGER DEFAULT 0,  -- 是否已有筹码分析结果
            avg_cost         REAL,               -- 市场平均成本
            profit_ratio     REAL,               -- 收盘获利比例（%）
            lhb_net          REAL,               -- 龙虎榜净买入额（万元，净卖出为负）
            main_net_today   REAL,               -- 今日主力（大单）净流入（亿元，净流出为负）
            main_net_5d      REAL,               -- 近5日主力净流入（亿元，净流出为负）
            margin_chg       REAL,               -- 融资融券余额较上一交易日变化（%）
            is_override      INTEGER DEFAULT 0,  -- 是否人类强制收编（1=无视AI拒绝、手动纳入）
            updated_at       TEXT                -- 最后更新时间
        )
        """
    )
    # 轻量迁移：给"老数据库"补上后来新增的列（列已存在会报错，忽略即可）
    for col, typ in [("avg_cost", "REAL"), ("profit_ratio", "REAL"), ("lhb_net", "REAL"),
                     ("main_net_today", "REAL"), ("main_net_5d", "REAL"), ("margin_chg", "REAL"),
                     ("is_override", "INTEGER DEFAULT 0"), ("eps", "REAL"),
                     ("serenity_score", "REAL"), ("score_detail", "TEXT"),
                     ("gdhs", "REAL"), ("gdhs_chg", "REAL"),
                     ("turnover", "REAL"), ("vol_ratio", "REAL")]:
        try:
            cur.execute(f"ALTER TABLE stock_pool ADD COLUMN {col} {typ}")
        except Exception:
            pass
    # 个人设置表（用于记住该用户的 API Key、模型名等，键值对存储）
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            k TEXT PRIMARY KEY,   -- 设置项名称
            v TEXT                -- 设置项的值
        )
        """
    )
    conn.commit()
    conn.close()


def save_setting(key, value):
    """保存（或覆盖）一个个人设置项到当前用户专属数据库。"""
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO app_settings (k, v) VALUES (?, ?) "
        "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (key, "" if value is None else str(value)),
    )
    conn.commit()
    conn.close()


def load_setting(key, default=""):
    """读取一个个人设置项；不存在时返回 default。"""
    conn = db_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT v FROM app_settings WHERE k=?", (key,))
        row = cur.fetchone()
    except Exception:
        row = None
    conn.close()
    return row[0] if row and row[0] is not None else default


def db_conn():
    """返回数据库连接（行可按列名访问）。库文件按登录用户隔离。"""
    conn = sqlite3.connect(current_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def load_pool_df():
    """读取整个股票池为 DataFrame。"""
    conn = db_conn()
    try:
        df = pd.read_sql_query("SELECT * FROM stock_pool", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df


def upsert_stock(code, name, is_perilla, analysis):
    """守门员通过后，插入或更新一只股票的基本信息。"""
    conn = db_conn()
    cur = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 已存在则只更新研判结果，保留行情/筹码/持仓等字段
    cur.execute("SELECT code FROM stock_pool WHERE code=?", (code,))
    if cur.fetchone():
        cur.execute(
            "UPDATE stock_pool SET name=?, is_perilla_leaf=?, analysis=?, updated_at=? WHERE code=?",
            (name, 1 if is_perilla else 0, analysis, now, code),
        )
    else:
        cur.execute(
            "INSERT INTO stock_pool (code, name, is_perilla_leaf, analysis, updated_at) VALUES (?,?,?,?,?)",
            (code, name, 1 if is_perilla else 0, analysis, now),
        )
    conn.commit()
    conn.close()


def force_add_stock(code, name, analysis):
    """
    👑 强制收编（上帝模式）：无视守门员的拒绝结论，强行把股票写入池子。
    关键点：把 is_perilla_leaf 置 1，这样它能像普通入池股一样，
    无障碍进入模块2(数据拉取)/模块3(筹码分析)/模块4(买卖决策)。
    同时打上 is_override=1 标记，前端会显示"人类强制加入"。
    """
    conn = db_conn()
    cur = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    note = ("【👑 人类强制收编】此股 AI 守门员判定『不符合紫苏叶标准』，"
            "由用户手动强行纳入池子。AI 原始拒绝理由：" + (analysis or "（无）"))
    cur.execute("SELECT code FROM stock_pool WHERE code=?", (code,))
    if cur.fetchone():
        cur.execute(
            "UPDATE stock_pool SET name=?, is_perilla_leaf=1, is_override=1, analysis=?, updated_at=? WHERE code=?",
            (name, note, now, code),
        )
    else:
        cur.execute(
            "INSERT INTO stock_pool (code, name, is_perilla_leaf, is_override, analysis, updated_at) "
            "VALUES (?, ?, 1, 1, ?, ?)",
            (code, name, note, now),
        )
    conn.commit()
    conn.close()


def update_fields(code, fields: dict):
    """通用字段更新：fields 为 {列名: 值}。"""
    if not fields:
        return
    fields = dict(fields)
    fields["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cols = ", ".join([f"{k}=?" for k in fields.keys()])
    vals = list(fields.values()) + [code]
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(f"UPDATE stock_pool SET {cols} WHERE code=?", vals)
    conn.commit()
    conn.close()


def delete_stock(code):
    """从股票池删除一只股票。"""
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM stock_pool WHERE code=?", (code,))
    conn.commit()
    conn.close()


def toggle_holding(code, is_holding):
    """设置持仓标记。"""
    update_fields(code, {"is_holding": 1 if is_holding else 0})


# ============================================================================
# 模块1：紫苏叶守门员（DeepSeek 基本面研判）
# ============================================================================

# ----------------------------------------------------------------------------
# 借鉴 serenity-skill（供应链瓶颈/卡脖子方法论）的两段共享研判框架。
# 注入到各个研判提示词里，让 AI 不只答"是/否"，而要说清"卡在第几层、命中哪些稀缺特征"。
# ----------------------------------------------------------------------------
SERENITY_FRAMEWORK = """【产业链 8 层分解】请把锚点业务定位到下面某一层（越靠上游、越底层，往往越稀缺）：
1.下游需求（终端应用/品牌） 2.系统集成（整机/方案） 3.模组子系统 4.芯片/器件
5.工艺/封装 6.设备/测试 7.材料/耗材 8.基础设施（电力、散热、产能等）
注意：不要被"AI芯片""新能源"这种大筐迷惑，要拆细——比如AI算力要拆成算力芯片/存储/EDA与IP/光模块/PCB与覆铜板/电源等，分别看谁更卡脖子。

【卡脖子 9 大特征】命中越多越稀缺、越像紫苏叶（请在理由里点明命中了哪几条）：
①供应商家数极少 ②客户认证/导入周期长 ③扩产难、经济性差 ④独家工艺know-how
⑤材料纯度/良率要求极高 ⑥强依赖专用设备 ⑦客户认证壁垒高、粘性强 ⑧交货周期长 ⑨产能需提前锁定/预定"""

# 8 维度评分 + 风险扣分的说明（注入到要求 AI 打分的提示词里）
SERENITY_SCORE_GUIDE = """【请给 8 个维度各打 0~5 分（整数）】含义：
需求拐点=下游需求是否正在加速放量；架构耦合=该环节是否被新技术架构强绑定、绕不开；
卡脖子严重度=供给有多紧、多难替代；供应商集中度=有效玩家是否极少；扩产难度=想扩产有多难；
证据质量=支撑判断的公开证据是否扎实；估值偏离=当前估值相对基本面是否便宜；催化时机=未来3~12月是否有明确催化。
【再给 4 项风险各打 0~5 分（整数，越高越糟，会扣分）】：增发摊薄、公司治理、炒作过热、财务质量。
打分原则：没把握就给 2~3 分，不要动不动给满分。"""

# 伪概念/假紫苏叶排除清单（注入各研判提示词，命中越多越要往否决/降分靠）
SERENITY_ANTIPATTERNS = """【伪概念排除 6 条——命中任一条都要警惕，命中越多越不像真紫苏叶，请在理由里点名是哪几条】：
①蹭热点贴标签：相关业务不是主营、占营收很小，只是为了蹭概念在互动平台/公告里贴标签；
②竞争分散没护城河：国内同行一大把、谁都能做，价格战激烈，没有真正壁垒；
③扩产太容易：产能想扩就扩、设备随便买、新玩家一两年就能进来，卡不住别人；
④"国产替代"伪命题：其实早已大面积国产化、或这环节根本不卡脖子，替代故事是讲给散户听的；
⑤利好已被充分定价：逻辑全市场都知道、股价已经炒高，估值透支了未来好几年的预期；
⑥无机构覆盖/流动性差：几乎没有券商研报覆盖、日成交额常年低于5000万，容易被操纵、买卖难成交。"""

PERILLA_SYSTEM_PROMPT = """你是一位顶级的A股硬科技产业链研究专家，精通"紫苏叶理论"。
"紫苏叶公司"的严格标准（必须同时满足）：
1. 处于产业链的深层节点（Layer3 及以下，即底层硬件、核心材料、关键设备等"卖水人"角色），而非终端品牌或应用层。
2. 产品/技术不可替代，具有很高的技术壁垒或专利护城河。
3. 处于寡头垄断格局（全球或国内有效竞争对手 <= 3 家）。

""" + SERENITY_FRAMEWORK + """

""" + SERENITY_SCORE_GUIDE + """

""" + SERENITY_ANTIPATTERNS + """

请基于你的知识，判断用户给出的公司是否符合"紫苏叶公司"标准。命中伪概念排除条目越多，越应下调 is_perilla_leaf 与评分。
你必须只返回一个 JSON 对象，不要任何额外文字、不要markdown代码块标记，格式严格如下：
{"is_perilla_leaf": true 或 false, "name": "公司中文简称", "chain_layer": "锚点业务卡在第几层（如：第4层 芯片/器件）", "antipattern_hits": "命中的伪概念排除条目（如：①蹭热点、⑤已被定价；没有则填 无）", "analysis": "用通俗易懂的大白话解释判断理由，点明卡在第几层、命中了哪几条卡脖子特征，若命中伪概念也要说清，150字以内，让股票小白也能看懂", "factors": {"需求拐点":0-5, "架构耦合":0-5, "卡脖子严重度":0-5, "供应商集中度":0-5, "扩产难度":0-5, "证据质量":0-5, "估值偏离":0-5, "催化时机":0-5}, "penalties": {"增发摊薄":0-5, "公司治理":0-5, "炒作过热":0-5, "财务质量":0-5}}
"""


def _deepseek_post(payload, api_key, timeout=DEEPSEEK_TIMEOUT, retries=DEEPSEEK_RETRIES):
    """
    统一的 DeepSeek 请求封装：带较长读取超时 + 超时自动重试。
    成功返回 (True, content_str, "")；失败返回 (False, None, err_msg)。
    """
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_err = ""
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                return False, None, f"DeepSeek 接口返回错误 {resp.status_code}：{resp.text[:200]}"
            content = resp.json()["choices"][0]["message"]["content"]
            return True, content, ""
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            # 超时或连接抖动：还有机会就重试，否则报告
            last_err = ("调用 DeepSeek 超时" if isinstance(e, requests.exceptions.Timeout)
                        else "连接 DeepSeek 失败")
            if attempt < retries:
                continue
            return False, None, f"{last_err}（已重试{retries}次）。请检查网络后再试，或稍后重试。"
        except requests.exceptions.RequestException as e:
            return False, None, f"网络请求异常：{e}"
        except Exception as e:
            return False, None, f"未知错误：{e}"
    return False, None, last_err or "调用 DeepSeek 失败。"


def call_deepseek_gatekeeper(api_key, model, user_input):
    """
    调用 DeepSeek 进行紫苏叶研判。
    返回 (ok, result_dict, err_msg)。
    result_dict 含 is_perilla_leaf / name / analysis。
    """
    if not api_key:
        return False, None, "未填写 DeepSeek API Key（请在左侧边栏填写）。"

    payload = {
        "model": model or "deepseek-chat",
        "messages": [
            {"role": "system", "content": PERILLA_SYSTEM_PROMPT},
            {"role": "user", "content": f"请研判这家公司：{user_input}"},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    ok, content, err = _deepseek_post(payload, api_key)
    if not ok:
        return False, None, err
    data = _safe_parse_json(content)
    if data is None or "is_perilla_leaf" not in data:
        return False, None, f"AI 返回内容无法解析为标准结果：{content[:200]}"
    return True, data, ""


PERILLA_AGENT_PROMPT = """你是一位顶级的A股硬科技产业链投研 Agent，精通"紫苏叶理论"，并以多轮对话的方式和用户协作选股。

【紫苏叶公司标准】（核心锚点业务需同时满足）：
1. 处于产业链深层节点（Layer3 及以下：底层硬件、核心材料、关键设备等"卖水人"），而非终端品牌或应用层。
2. 产品/技术不可替代，技术壁垒或专利护城河高。
3. 寡头垄断格局（全球或国内有效竞争对手 <= 3 家）。

""" + SERENITY_FRAMEWORK + """

【你的工作方式——非常重要】：
不要拿一家公司的"总盘子业务"去一刀切地否定它。很多公司主业是红海（如汽车齿轮、消费电子组装），
但其内部往往藏着一条符合紫苏叶特征的"核心零部件/隐藏业务"。你必须：
1. 先在脑中拆解该公司的全部业务线；
2. 主动过滤掉竞争激烈的红海业务；
3. 主动挖掘其中最可能符合紫苏叶特征的细分业务/关键零部件，作为"紫苏叶锚点"；
4. 然后用大白话向用户说明："主业XX是红海不达标，但发现隐藏的细分业务YY处于产业链底层、玩家极少，是否以YY作为紫苏叶锚点深度评估并入池？"
5. 与用户探讨。用户若同意，就以该锚点入库；用户若说"逻辑不硬"或指出别的赛道，你要据其指示重新评估、另找锚点。
6. 若整家公司确实找不到任何站得住脚的紫苏叶锚点，要如实告知，不要硬凑。

【输出格式——必须严格遵守】：
你每一轮都只返回一个 JSON 对象（不要任何额外文字、不要markdown代码块标记），字段如下：
{
 "reply": "要显示在聊天框里的大白话内容（这是用户唯一能看到的文字，要自然、口语化、像投研伙伴在跟小白聊天）",
 "ready_to_add": true 或 false,   // 仅当『用户已明确同意以某锚点入池』时才为 true；否则一律 false
 "code": "6位股票代码或 null",
 "name": "公司中文简称或 null",
 "anchor": "确认入池时的紫苏叶锚点业务名称（如 RV减速器）或 null",
 "analysis": "确认入池时，用大白话总结『为什么锚定该细分业务它就算紫苏叶』，150字内；未入池则 null",
 "chain_layer": "确认入池时，锚点业务卡在第几层（如：第4层 芯片/器件）；未入池则 null",
 "factors": 确认入池时给出 {"需求拐点":0-5,"架构耦合":0-5,"卡脖子严重度":0-5,"供应商集中度":0-5,"扩产难度":0-5,"证据质量":0-5,"估值偏离":0-5,"催化时机":0-5}；未入池则 null,
 "penalties": 确认入池时给出 {"增发摊薄":0-5,"公司治理":0-5,"炒作过热":0-5,"财务质量":0-5}；未入池则 null
}

""" + SERENITY_SCORE_GUIDE + """
规则：
- 第一次收到一个公司时，先做拆解+提出锚点建议并询问，ready_to_add 必须为 false。
- 只有当最近一条用户消息表达了同意（如"同意""就按这个逻辑来""可以""加进去"）时，ready_to_add 才设为 true，并把 code/name/anchor/analysis/chain_layer/factors/penalties 填全。
- 用户否定或提出新方向时，ready_to_add 为 false，在 reply 里重新评估。
"""


def call_deepseek_agent(api_key, model, history):
    """
    多轮对话投研 Agent。history 为 [{"role":"user"/"assistant","content":...}] 列表。
    返回 (ok, result_dict, err_msg)。result_dict 含 reply/ready_to_add/code/name/anchor/analysis。
    """
    if not api_key:
        return False, None, "未填写 DeepSeek API Key（请在左侧边栏填写）。"
    messages = [{"role": "system", "content": PERILLA_AGENT_PROMPT}]
    messages.extend(history)
    payload = {
        "model": model or "deepseek-chat",
        "messages": messages,
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    ok, content, err = _deepseek_post(payload, api_key)
    if not ok:
        return False, None, err
    data = _safe_parse_json(content)
    if data is None or "reply" not in data:
        return False, None, f"AI 返回内容无法解析为标准结果：{content[:200]}"
    return True, data, ""


PERILLA_MINER_PROMPT = """你是一位顶级的A股硬科技产业链投研专家，精通"紫苏叶理论"，现在要主动跨赛道为用户挖掘潜力股。

【紫苏叶公司标准】（推荐的每只股需同时满足）：
1. 处于产业链深层节点（Layer3 及以下：底层硬件、核心材料、关键设备、卡脖子零部件等"卖水人"），而非终端品牌或应用层。
2. 产品/技术不可替代，技术壁垒或专利护城河高。
3. 寡头垄断格局（全球或国内有效竞争对手 <= 3 家）。

""" + SERENITY_FRAMEWORK + """

""" + SERENITY_SCORE_GUIDE + """

""" + SERENITY_ANTIPATTERNS + """

【A股卡脖子样例参考】（只是示范"卡在产业链深层的卖水人"是什么范式，帮助你给出准确的细分环节与真实代码；不要只局限于这些，要按当下热度灵活发挥）：
· 电力/特高压 → 换流变压器、套管等核心设备（如 国电南瑞 600406）
· 光通信/AI算力 → 上游 EML/DFB 激光器芯片（如 源杰科技 688498）
· 半导体设备 → 薄膜沉积/刻蚀等关键设备（如 中微公司 688012、北方华创 002371）
· 军工新材料 → 高温特种合金、碳纤维（如 中简科技 300777、光威复材 300699）
· 新能源车 → 碳纳米管导电剂（如 天奈科技 688116）、高镍正极（如 当升科技 300073）
· 创新药 → 自主靶点 + CDMO 产能（如 恒瑞医药 600276、药明康德 603259）
· 种业 → 转基因种子性状（如 隆平高科 000998、大北农 002385）
要点：这些都是"别人绕不开、玩家极少、扩产难"的深层环节；挖掘时找类似卡位的细分龙头，别被"AI/机器人/新能源"等大筐迷惑，务必拆到最卡脖子的那一层。

【任务——三步思维链】：
第一步：识别当前A股市场最核心的 3 到 5 个硬科技热门赛道（要多元，不要只盯机器人；可考虑如固态电池、低空经济、商业航天、合成生物、AI算力/光模块、半导体设备/材料、可控核聚变等当下真实热门方向）。
第二步：在每个赛道里，用紫苏叶理论深度挖掘那条"别人离不开、卡脖子、玩家极少"的底层环节；并用伪概念排除 6 条自查、剔除蹭热点的假票。
第三步：每个赛道推荐 1-2 只最符合紫苏叶标准的 A 股上市公司（须是真实存在的A股，给出准确的6位代码）。

【输出格式——必须严格遵守】：
只返回一个 JSON 对象（不要任何额外文字、不要markdown代码块标记），格式如下：
{
 "sectors": [
   {
     "sector": "赛道名称（如：固态电池）",
     "logic": "这个赛道里紫苏叶环节在哪、为什么是卖水人（大白话，60字内）",
     "stocks": [
       {
         "name": "公司中文简称",
         "code": "6位股票代码",
         "chain_layer": "卡在产业链第几层（如：第4层 芯片/器件）",
         "bottleneck": "它卡的是哪个脖子/处在哪个底层节点（大白话），并点明命中哪几条卡脖子特征",
         "competitors": "全球或国内的有效竞争对手大致有哪几家（体现玩家极少）",
         "reason": "用大白话总结为什么它符合紫苏叶标准，100字内，让股票小白也能看懂",
         "thesis_breaker": "⚠️这套逻辑的死穴：什么情况一旦发生，就说明看错了/该回避（如：被某新技术替代、对手扩产、客户自研），大白话，60字内",
         "antipattern_hits": "命中的伪概念排除条目（如：①蹭热点、③扩产太容易；没有则填 无）",
         "confidence": "证据可信度，只能填三选一：已确认 / 推断 / 待核实（'已确认'=公认事实；'推断'=你的合理推断；'待核实'=不太确定）",
         "factors": {"需求拐点":0-5, "架构耦合":0-5, "卡脖子严重度":0-5, "供应商集中度":0-5, "扩产难度":0-5, "证据质量":0-5, "估值偏离":0-5, "催化时机":0-5},
         "penalties": {"增发摊薄":0-5, "公司治理":0-5, "炒作过热":0-5, "财务质量":0-5}
       }
     ]
   }
 ]
}
注意：宁缺毋滥，拿不准是否真实存在的公司不要硬编代码；代码必须是真实的6位A股代码。"""


def call_deepseek_miner(api_key, model):
    """
    AI 主动跨赛道挖掘紫苏叶股。返回 (ok, result_dict, err_msg)。
    result_dict 含 sectors 列表。
    """
    if not api_key:
        return False, None, "未填写 DeepSeek API Key（请在左侧边栏填写）。"
    payload = {
        "model": model or "deepseek-chat",
        "messages": [
            {"role": "system", "content": PERILLA_MINER_PROMPT},
            {"role": "user", "content": "请现在扫描全市场热门硬科技赛道，按要求挖掘并推荐紫苏叶股。"},
        ],
        "temperature": 0.6,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    # 思维链较长，给更宽的读取超时
    ok, content, err = _deepseek_post(payload, api_key, timeout=(15, 240))
    if not ok:
        return False, None, err
    data = _safe_parse_json(content)
    if data is None or "sectors" not in data:
        return False, None, f"AI 返回内容无法解析为标准结果：{content[:200]}"
    return True, data, ""


def _safe_parse_json(text):
    """从模型返回文本中尽量稳妥地解析出 JSON 对象。"""
    if not text:
        return None
    text = text.strip()
    # 去掉可能的 markdown 代码块包裹
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except Exception:
        pass
    # 兜底：截取第一个 { 到最后一个 }
    try:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e + 1])
    except Exception:
        return None
    return None


# ============================================================================
# 模块2：动态数据拉取（akshare 行情/财务/龙虎榜）
# ============================================================================

def _get_akshare():
    """延迟导入 akshare，避免环境未装时整页崩溃。"""
    try:
        import akshare as ak
        return ak
    except Exception:
        return None


def _prefixed_symbol(code):
    """6位代码 → 带市场前缀的代码（新浪/腾讯接口需要，如 sz002472 / sh600519 / bj830799）。"""
    c = str(code).zfill(6)
    if c[0] == "6":
        return "sh" + c           # 沪市主板/科创板(688)
    if c[0] in ("0", "3"):
        return "sz" + c           # 深市主板/创业板(300)
    if c[0] in ("4", "8"):
        return "bj" + c           # 北交所
    if c[0] == "9":
        return "sh" + c           # 沪市B股
    return "sz" + c


def _ma_from_close(df, date_col, close_col):
    """从含『日期+收盘』的日线表计算 close 与 MA10/20/30；数据不足返回 None。"""
    if df is None or len(df) == 0 or close_col not in df.columns:
        return None
    d = df.copy()
    if date_col in d.columns:
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d = d.dropna(subset=[date_col]).sort_values(date_col)
    close = pd.to_numeric(d[close_col], errors="coerce").dropna()
    if close.empty:
        return None

    def ma(n):
        return round(float(close.rolling(n).mean().iloc[-1]), 2) if len(close) >= n else None

    return {"close": round(float(close.iloc[-1]), 2),
            "ma10": ma(10), "ma20": ma(20), "ma30": ma(30)}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_price_ma(code):
    """
    拉取日线行情，计算最新收盘价与 MA10/MA20/MA30。
    多数据源依次尝试：新浪 → 腾讯 → 东方财富。
    （东方财富的行情/快照服务器对部分网络、云服务器 IP 会拒绝连接，故优先用新浪/腾讯。）
    成功返回 dict；全部失败则抛出异常（Streamlit 不缓存异常，下次刷新会自动重试）。
    """
    ak = _get_akshare()
    if ak is None:
        raise RuntimeError("akshare 未安装")
    sym = _prefixed_symbol(code)
    last_err = "未知原因"

    # 源1：新浪（最稳、最快）
    try:
        df = ak.stock_zh_a_daily(symbol=sym, adjust="qfq")
        r = _ma_from_close(df, "date", "close")
        if r:
            return r
        last_err = "新浪返回空数据"
    except Exception as e:
        last_err = f"新浪源失败：{e}"

    # 源2：腾讯
    try:
        df = ak.stock_zh_a_hist_tx(symbol=sym, adjust="qfq")
        r = _ma_from_close(df, "date", "close")
        if r:
            return r
        last_err = "腾讯返回空数据"
    except Exception as e:
        last_err = f"腾讯源失败：{e}"

    # 源3：东方财富（最后兜底，部分网络下会连接被拒）
    try:
        end = datetime.date.today().strftime("%Y%m%d")
        start = (datetime.date.today() - datetime.timedelta(days=400)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=str(code).zfill(6), period="daily",
                                start_date=start, end_date=end, adjust="qfq")
        r = _ma_from_close(df, "日期", "收盘")
        if r:
            return r
        last_err = "东财返回空数据"
    except Exception as e:
        last_err = f"东财源失败：{e}"

    raise RuntimeError(f"行情数据获取失败（已尝试新浪/腾讯/东财）：{last_err}")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_npr_growth(code):
    """
    拉取『净利润同比增长率（%）』与『基本每股收益 EPS』。
    返回二元组 (npr_growth, eps)；任一取不到则该项为 None，全失败返回 (None, None)。

    数据源优先级：
      主源：同花顺网页财务摘要 stock_financial_abstract_ths（含真实 EPS，更贴合需求口径）
      备源1：东方财富财务摘要 stock_financial_abstract（仅净利润同比，EPS 取不到给 None）
      备源2：东方财富业绩报表 stock_yjbb_em（仅净利润同比）
    """
    ak = _get_akshare()
    if ak is None:
        return None, None
    c = str(code).zfill(6)

    # ---- 主源：同花顺网页（行序为旧→新，最后一行=最新报告期）----
    try:
        df = ak.stock_financial_abstract_ths(symbol=c, indicator="按报告期")
        if df is not None and not df.empty:
            npr = None
            eps = None
            # 净利润同比增长率：从最后一行（最新）往前找首个可解析值
            if "净利润同比增长率" in df.columns:
                for v in reversed(list(df["净利润同比增长率"].values)):
                    val = _to_float_pct(v)
                    if val is not None:
                        npr = val
                        break
            # 基本每股收益：取最新一行（最后一行）
            if "基本每股收益" in df.columns:
                eps = _to_float_pct(df["基本每股收益"].iloc[-1])
            if npr is not None or eps is not None:
                return npr, eps
    except Exception:
        pass

    # ---- 备源1：东方财富财务摘要（仅净利润同比）----
    try:
        df = ak.stock_financial_abstract(symbol=code)
        if df is not None and not df.empty and "指标" in df.columns:
            mask = df["指标"].astype(str).str.contains("净利润") & \
                   df["指标"].astype(str).str.contains("同比|增长")
            sub = df[mask]
            if not sub.empty:
                row = sub.iloc[0]
                for v in reversed(list(row.values)):
                    val = _to_float_pct(v)
                    if val is not None:
                        return val, None
    except Exception:
        pass

    # ---- 备源2：东方财富业绩报表（仅净利润同比）----
    try:
        year = datetime.date.today().year
        for q_date in [f"{year}0331", f"{year-1}1231", f"{year-1}0930"]:
            try:
                df = ak.stock_yjbb_em(date=q_date)
            except Exception:
                continue
            if df is None or df.empty:
                continue
            code_col = "股票代码" if "股票代码" in df.columns else df.columns[1]
            row = df[df[code_col].astype(str).str.zfill(6) == c]
            if not row.empty:
                for col in df.columns:
                    if "净利润" in str(col) and ("同比" in str(col) or "增长" in str(col)):
                        val = _to_float_pct(row.iloc[0][col])
                        if val is not None:
                            return val, None
    except Exception:
        pass

    return None, None


def _pe_pct_from_series(df, date_col, pe_col):
    """从含『日期 + PE』的序列计算当前 PE 与近3年分位。返回 (pe, pct) 或 (None, None)。"""
    try:
        if df is None or len(df) == 0 or pe_col not in df.columns:
            return None, None
        d = df[[date_col, pe_col]].copy()
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d[pe_col] = pd.to_numeric(d[pe_col], errors="coerce")
        d = d.dropna().sort_values(date_col)
        if d.empty:
            return None, None
        cutoff = pd.Timestamp(datetime.date.today() - datetime.timedelta(days=365 * 3))
        recent = d[d[date_col] >= cutoff]
        if recent.empty:
            recent = d
        s = recent[pe_col].astype(float)
        cur_pe = float(s.iloc[-1])
        pct = round(float((s <= cur_pe).mean() * 100), 1)  # 当前 PE 在近3年中的百分位
        return round(cur_pe, 2), pct
    except Exception:
        return None, None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pe_percentile(code):
    """
    拉取历史 PE(TTM) 序列，计算当前 PE 及其近3年分位（%）。
    数据源：东方财富个股估值 stock_value_em（主）→ 百度股市通 stock_zh_valuation_baidu（备）。
    （注：旧版用的 stock_a_indicator_lg 在新版 akshare 已被移除，这是之前 PE 总取不到的主因。）
    返回 (pe, pe_percentile)；失败返回 (None, None)。
    """
    ak = _get_akshare()
    if ak is None:
        return None, None
    c = str(code).zfill(6)

    # 源1：东方财富个股估值（含 PE(TTM) 日序列，可同时算现值与分位）
    try:
        df = ak.stock_value_em(symbol=c)
        if df is not None and not df.empty:
            date_col = "数据日期" if "数据日期" in df.columns else df.columns[0]
            pe_col = None
            for cand in ("PE(TTM)", "PE（TTM）", "市盈率(TTM)"):
                if cand in df.columns:
                    pe_col = cand
                    break
            if pe_col:
                r = _pe_pct_from_series(df, date_col, pe_col)
                if r != (None, None):
                    return r
    except Exception:
        pass

    # 源2：百度股市通 市盈率(TTM)
    try:
        df = ak.stock_zh_valuation_baidu(symbol=c, indicator="市盈率(TTM)", period="近五年")
        if df is not None and not df.empty and "value" in df.columns:
            r = _pe_pct_from_series(df, "date", "value")
            if r != (None, None):
                return r
    except Exception:
        pass

    return None, None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_lhb_flag(code):
    """判断该股近一交易日是否上龙虎榜。返回 True/False/None。"""
    ak = _get_akshare()
    if ak is None:
        return None
    try:
        end = datetime.date.today().strftime("%Y%m%d")
        start = (datetime.date.today() - datetime.timedelta(days=10)).strftime("%Y%m%d")
        df = ak.stock_lhb_detail_em(start_date=start, end_date=end)
        if df is None or df.empty:
            return False
        code_col = "代码" if "代码" in df.columns else ("股票代码" if "股票代码" in df.columns else df.columns[0])
        hit = df[df[code_col].astype(str).str.zfill(6) == code]
        return not hit.empty
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_chip_cyq(code):
    """
    【自动筹码】用东方财富『筹码分布』接口 stock_cyq_em 自动获取筹码数据，
    免去每次都要手动上传筹码图。返回 dict；抓不到返回 None。

    返回字段：
      chip_single_peak / chip_above_avg / chip_high_diverge（三个布尔，1/0）
      profit_ratio（收盘获利比例%）、avg_cost（市场平均成本）、chip_confidence（≈0.6）

    重要说明：
      1) 这三个布尔是【程序启发式估算】，没有"看图"那么精准，仅作默认值；
         你仍可到『上传筹码图』页面用视觉模型/人工修正得到更准的结论（会优先生效）。
      2) 东财筹码服务器对部分网络/云服务器 IP 可能拒绝连接；失败时本函数返回 None，
         上层会保留原值并回退到"上传筹码图"老路，不影响股价/财务等其他数据。
    """
    ak = _get_akshare()
    if ak is None:
        return None
    c = str(code).zfill(6)
    try:
        df = ak.stock_cyq_em(symbol=c, adjust="qfq")
    except Exception:
        return None
    if df is None or getattr(df, "empty", True):
        return None
    try:
        date_col = "日期" if "日期" in df.columns else df.columns[0]
        try:
            df = df.sort_values(date_col)
        except Exception:
            pass
        last = df.iloc[-1]

        def _g(*names):
            for n in names:
                if n in df.columns:
                    try:
                        fv = float(last.get(n))
                        if fv == fv:  # 排除 NaN
                            return fv
                    except Exception:
                        continue
            return None

        profit = _g("获利比例")          # 0~1，越大说明越多筹码处于盈利
        avg_cost = _g("平均成本")
        conc90 = _g("90集中度")          # 越小=筹码越集中（单峰）；越大=越分散
        # 关键字段都没有就放弃
        if profit is None and avg_cost is None and conc90 is None:
            return None

        # —— 启发式派生三布尔 ——
        # 低位单峰密集：筹码很集中(90集中度小) 且 价格不在高位(获利盘不算太多)
        single_peak = bool(
            conc90 is not None and conc90 <= 0.15
            and (profit is None or profit <= 0.60)
        )
        # 站上平均成本线：超过半数筹码处于盈利 ≈ 现价在平均成本之上
        above_avg = bool(profit is not None and profit >= 0.50)
        # 高位发散：几乎全员获利 且 筹码偏分散
        high_diverge = bool(
            profit is not None and profit >= 0.90
            and (conc90 is None or conc90 >= 0.20)
        )

        return {
            "chip_single_peak": 1 if single_peak else 0,
            "chip_above_avg": 1 if above_avg else 0,
            "chip_high_diverge": 1 if high_diverge else 0,
            "profit_ratio": round(profit * 100, 2) if profit is not None else None,
            "avg_cost": round(avg_cost, 2) if avg_cost is not None else None,
            "chip_confidence": 0.6,
        }
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_gdhs(code):
    """
    【股东户数】东方财富 stock_zh_a_gdhs_detail_em（按个股，返回历年股东户数明细）。
    返回 (最新股东户数, 较上期变化%)；失败返回 (None, None)。
    口径：户数变化为负（减少）= 筹码在集中 = 利好；为正（增加）= 筹码在分散 = 偏空。
    （注意：不要用 stock_zh_a_gdhs，那个是按季度日期取全市场、会遍历上千只、极慢。）
    """
    ak = _get_akshare()
    if ak is None:
        return None, None
    c = str(code).zfill(6)
    try:
        df = ak.stock_zh_a_gdhs_detail_em(symbol=c)
    except Exception:
        return None, None
    if df is None or getattr(df, "empty", True):
        return None, None

    def _num(v):
        try:
            fv = float(str(v).replace(",", "").replace("%", "").strip())
            return fv if fv == fv else None
        except Exception:
            return None

    try:
        # 按截止日排序，取最新一行
        date_col = None
        for cand in ("股东户数统计截止日", "截止日", "股东户数公告日期", "报告期"):
            if cand in df.columns:
                date_col = cand
                break
        if date_col:
            try:
                df = df.sort_values(date_col)
            except Exception:
                pass
        last = df.iloc[-1]

        latest = None
        for cand in ("股东户数-本次", "股东户数", "期末股东户数"):
            if cand in df.columns:
                latest = _num(last.get(cand))
                if latest is not None:
                    break

        chg = None
        # 优先用现成的"增减比例"列
        for cand in ("股东户数-增减比例", "增减比例"):
            if cand in df.columns:
                chg = _num(last.get(cand))
                if chg is not None:
                    break
        # 没有现成比例则用 本次/上次 计算
        if chg is None:
            prev = None
            for cand in ("股东户数-上次", "上次股东户数"):
                if cand in df.columns:
                    prev = _num(last.get(cand))
                    if prev is not None:
                        break
            if prev is None:
                # 退而求其次：用倒数第二行的户数
                vals = []
                for cand in ("股东户数-本次", "股东户数", "期末股东户数"):
                    if cand in df.columns:
                        vals = [_num(x) for x in df[cand].tolist()]
                        vals = [x for x in vals if x is not None]
                        break
                if len(vals) >= 2:
                    prev = vals[-2]
                    latest = latest if latest is not None else vals[-1]
            if prev and latest is not None and prev != 0:
                chg = round((latest - prev) / prev * 100, 2)

        if latest is None:
            return None, None
        return latest, chg
    except Exception:
        return None, None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_turnover_vol(code):
    """
    【换手率 & 量比】仅用东财日线接口（stock_zh_a_hist）拉取。
    返回 (换手率%, 量比)；量比 = 最新成交量 / 近5日平均成交量（今日不含）。
    取不到任意项则对应值为 None；全失败返回 (None, None)。
    """
    ak = _get_akshare()
    if ak is None:
        return None, None
    try:
        end = datetime.date.today().strftime("%Y%m%d")
        start = (datetime.date.today() - datetime.timedelta(days=60)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=str(code).zfill(6), period="daily",
                                start_date=start, end_date=end, adjust="qfq")
        if df is None or len(df) < 6:
            return None, None
        if "日期" in df.columns:
            df = df.sort_values("日期")
        # 换手率
        turnover = None
        if "换手率" in df.columns:
            try:
                turnover = round(float(df["换手率"].iloc[-1]), 2)
            except Exception:
                pass
        # 量比 = 最新成交量 / 前5日均量
        vol_ratio = None
        if "成交量" in df.columns:
            try:
                vol = pd.to_numeric(df["成交量"], errors="coerce").dropna()
                if len(vol) >= 6:
                    today_vol = float(vol.iloc[-1])
                    avg5 = float(vol.iloc[-6:-1].mean())
                    if avg5 > 0:
                        vol_ratio = round(today_vol / avg5, 2)
            except Exception:
                pass
        return turnover, vol_ratio
    except Exception:
        return None, None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_recent_rally(code):
    """
    【近期涨幅】算该股近约 60 个交易日（~100 自然日）的累计涨幅%，用于『暴涨过滤·避免追高』。
    返回 dict：{"gain_pct": 累计涨幅%或None, "why": 大白话说明}。
    取不到时 gain_pct=None（按『不暴涨』处理，避免误杀），不抛异常。
    """
    ak = _get_akshare()
    if ak is None:
        return {"gain_pct": None, "why": "未安装行情库"}
    try:
        end = datetime.date.today().strftime("%Y%m%d")
        start = (datetime.date.today() - datetime.timedelta(days=100)).strftime("%Y%m%d")
        df = ak.stock_zh_a_hist(symbol=str(code).zfill(6), period="daily",
                                start_date=start, end_date=end, adjust="qfq")
        if df is None or len(df) < 20:
            return {"gain_pct": None, "why": "历史数据不足"}
        if "日期" in df.columns:
            df = df.sort_values("日期")
        closes = pd.to_numeric(df["收盘"], errors="coerce").dropna() if "收盘" in df.columns else None
        if closes is None or len(closes) < 20:
            return {"gain_pct": None, "why": "历史数据不足"}
        first = float(closes.iloc[0])
        last = float(closes.iloc[-1])
        if first <= 0:
            return {"gain_pct": None, "why": "数据异常"}
        gain_pct = round((last - first) / first * 100, 1)
        return {"gain_pct": gain_pct, "why": f"近2-3月累计涨幅 {gain_pct}%"}
    except Exception:
        return {"gain_pct": None, "why": "行情没拉到"}


def _pick_col(df, candidates):
    """从 DataFrame 里按候选名顺序找第一个存在的列名，找不到返回 None。"""
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fetch_hot_boards(top_n=3):
    """
    【热门板块】取当前涨幅最高的前 top_n 个概念板块（主源）/行业板块（备源）。
    返回 (boards:list[{"board":名,"pct":涨幅%}], err:str)。成功时 err=""；全失败时 boards=[] 且 err 含原因。
    不做缓存：本函数只在点按钮时触发，避免把临时失败缓存住导致『稍后再点也没用』。
    """
    ak = _get_akshare()
    if ak is None:
        return [], "未安装行情库 akshare"
    errs = []
    for fn in ("stock_board_concept_name_em", "stock_board_industry_name_em",
               "stock_board_concept_name_ths", "stock_board_industry_summary_ths"):
        func = getattr(ak, fn, None)
        if func is None:
            errs.append(f"{fn}:无此接口")
            continue
        try:
            df = func()
        except Exception as e:
            errs.append(f"{fn}:{repr(e)[:80]}")
            continue
        if df is None or df.empty:
            errs.append(f"{fn}:返回空表")
            continue
        name_col = _pick_col(df, ["板块名称", "概念名称", "行业名称", "板块", "名称"])
        pct_col = _pick_col(df, ["涨跌幅", "涨幅", "涨跌幅(%)"])
        if not name_col or not pct_col:
            errs.append(f"{fn}:列名不匹配({list(df.columns)[:6]})")
            continue
        try:
            tmp = df.copy()
            tmp["_pct"] = pd.to_numeric(tmp[pct_col].astype(str).str.replace("%", "", regex=False),
                                        errors="coerce")
            tmp = tmp.dropna(subset=["_pct"]).sort_values("_pct", ascending=False)
            out = []
            for _, r in tmp.head(top_n).iterrows():
                out.append({"board": str(r[name_col]), "pct": round(float(r["_pct"]), 2)})
            if out:
                return out, ""
            errs.append(f"{fn}:排序后为空")
        except Exception as e:
            errs.append(f"{fn}:{repr(e)[:80]}")
            continue
    return [], "；".join(errs) if errs else "未知原因"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_board_cons(board):
    """
    【板块成分股】取某板块的成分股精简表（代码/名称/今日涨跌幅/换手率/最新价）。
    主源概念成分，备源行业成分。返回 DataFrame（统一列名 code/name/pct/turnover/price）；失败返回 None。
    """
    ak = _get_akshare()
    if ak is None or not board:
        return None
    for fn in ("stock_board_industry_cons_ths", "stock_board_concept_cons_ths",
               "stock_board_concept_cons_em", "stock_board_industry_cons_em"):
        try:
            func = getattr(ak, fn, None)
            if func is None:
                continue
            # 同花顺接口用 sector=，东财接口用 symbol=，逐一尝试
            try:
                df = func(symbol=str(board))
            except TypeError:
                try:
                    df = func(sector=str(board))
                except Exception:
                    df = None
            if df is None or df.empty:
                continue
            code_col = _pick_col(df, ["代码", "股票代码", "code"])
            name_col = _pick_col(df, ["名称", "股票名称", "name"])
            pct_col = _pick_col(df, ["涨跌幅", "涨幅", "涨跌幅(%)"])
            to_col = _pick_col(df, ["换手率", "换手率(%)"])
            price_col = _pick_col(df, ["最新价", "现价", "收盘", "最新"])
            if not code_col or not name_col:
                continue
            out = pd.DataFrame()
            out["code"] = df[code_col].astype(str).str.zfill(6)
            out["name"] = df[name_col].astype(str)
            out["pct"] = pd.to_numeric(df[pct_col], errors="coerce") if pct_col else None
            out["turnover"] = pd.to_numeric(df[to_col], errors="coerce") if to_col else None
            out["price"] = pd.to_numeric(df[price_col], errors="coerce") if price_col else None
            return out
        except Exception:
            continue
    return None


def _to_float_pct(v):
    """把可能带 % 或文字的值转成 float；不可转返回 None。"""
    if v is None:
        return None
    try:
        s = str(v).replace("%", "").replace(",", "").strip()
        if s in ("", "--", "nan", "None", MISSING):
            return None
        return round(float(s), 2)
    except Exception:
        return None


def refresh_one_stock(code):
    """拉取并写入单只股票的全部行情/财务/龙虎榜数据，单项失败不影响其他项。"""
    fields = {}

    # 通用原则：akshare 取到值就用（更准、会覆盖）；取不到（None）就【不写入】，
    # 以免把筹码图读出的 / 你手动录入的数值清成空白。因抛异常未被缓存，下次刷新会自动重试。

    # 行情与均线
    try:
        ma = fetch_price_ma(code)
        for k in ("close", "ma10", "ma20", "ma30"):
            v = ma.get(k)
            if v is not None:
                fields[k] = v
    except Exception:
        pass

    # 净利润同比增长 + 基本每股收益 EPS（同花顺源）
    npr, eps = fetch_npr_growth(code)
    if npr is not None:
        fields["npr_growth"] = npr
    if eps is not None:
        fields["eps"] = eps

    # PE 与近3年分位
    pe, pe_pct = fetch_pe_percentile(code)
    if pe is not None:
        fields["pe"] = pe
    if pe_pct is not None:
        fields["pe_percentile"] = pe_pct

    # 龙虎榜（取不到则不动原值）
    lhb = fetch_lhb_flag(code)
    if lhb is not None:
        fields["lhb_flag"] = 1 if lhb else 0

    # 股东户数（户数减少=筹码集中=利好；取不到则不动原值）
    gdhs, gdhs_chg = fetch_gdhs(code)
    if gdhs is not None:
        fields["gdhs"] = gdhs
    if gdhs_chg is not None:
        fields["gdhs_chg"] = gdhs_chg

    # 换手率 & 量比（量价信号基础数据；仅东财源，取不到不动原值）
    turnover, vol_ratio = fetch_turnover_vol(code)
    if turnover is not None:
        fields["turnover"] = turnover
    if vol_ratio is not None:
        fields["vol_ratio"] = vol_ratio

    # 自动筹码（免上传图）：仅当该股【没有被人工修正过】时，才用自动估算覆盖筹码字段。
    # 人工/视觉看图的结论更准，必须优先保留（chip_manual=1 时跳过自动覆盖）。
    try:
        conn = db_conn()
        cur = conn.cursor()
        cur.execute("SELECT chip_manual FROM stock_pool WHERE code=?", (code,))
        _r = cur.fetchone()
        conn.close()
        manual = bool(_r["chip_manual"]) if _r is not None else False
    except Exception:
        manual = False
    if not manual:
        chip = fetch_chip_cyq(code)
        if chip:
            for k in ("chip_single_peak", "chip_above_avg", "chip_high_diverge",
                      "profit_ratio", "avg_cost", "chip_confidence"):
                v = chip.get(k)
                if v is not None:
                    fields[k] = v
            # 自动拿到筹码 → 标记 has_chip=1，这样无需上传图也能出完整买卖建议
            fields["has_chip"] = 1

    if fields:
        update_fields(code, fields)


# ============================================================================
# 模块3：AI 视觉分析师（Gemini 筹码分布图分析）
# ============================================================================

CHIP_VISION_PROMPT = """你是一位精通筹码分布（成本分布）与K线均线分析的A股技术专家。
我会给你一张同花顺风格的图（通常含筹码分布、现价、均线等信息）。请仔细看图后，只返回一个 JSON 对象，
不要任何额外文字、不要markdown标记。格式严格如下：
{
 "is_single_peak_low": true 或 false,   // 是否为"低位单峰密集"（筹码高度集中在当前价附近的低位区域）
 "above_avg_cost": true 或 false,       // 当前股价是否站上了"平均成本线"
 "is_high_diverge": true 或 false,      // 是否为"高位发散"（筹码在高位分散、获利盘巨大、有派发风险）
 "close_price": 数字 或 null,           // 图中显示的"最新价/现价/收盘价"，读不到就填 null
 "ma10": 数字 或 null,                  // 图中10日均线（MA10/M10）的当前数值，看不清就填 null
 "ma20": 数字 或 null,                  // 图中20日均线（MA20/M20）的当前数值，看不清就填 null
 "ma30": 数字 或 null,                  // 图中30日均线（MA30/M30）的当前数值，看不清就填 null
 "pe": 数字 或 null,                    // 【只读"市盈(TTM)"这一项】！图中通常同时有 市盈(动)、市盈(静)、市盈(TTM) 三个数，请务必只取"市盈TTM/市盈(TTM)/PE(TTM)"那个数值，不要拿动态或静态市盈率；找不到 TTM 就填 null
 "avg_cost": 数字 或 null,              // 图中筹码区的"平均成本"，读不到就填 null
 "profit_ratio": 数字 或 null,          // 图中"收盘获利比例/获利比例"的百分数（只要数字，如 100），读不到就填 null
 "lhb_net_wan": 数字 或 null,           // 图中"龙虎榜净买入额"，单位万元，净卖出为负数（如 -6803）；图上没有就填 null
 "main_net_today_yi": 数字 或 null,     // 图中"今日主力净流入/大单净流入/主力净额"，单位【亿元】，净流出为负数；图上没有就填 null
 "main_net_5d_yi": 数字 或 null,        // 图中"近5日/5日主力净流入"，单位【亿元】，净流出为负数；图上没有就填 null
 "margin_change_pct": 数字 或 null,     // 图中"融资融券余额"较上一交易日的变化百分比（只要数字，如 6.94 表示+6.94%；减少为负）；图上没有就填 null
 "confidence": 0.0 到 1.0 之间的小数,    // 你对本次判断的置信度
 "explain": "用大白话解释当前筹码与均线状态，100字以内，让股票小白能看懂"
}
注意：所有数字请尽量读取图中标注的真实数值；价格/均线若只有线没数字，可结合纵轴价格刻度估算；实在判断不了才填 null。
龙虎榜与资金金额注意正负号：净卖出/净流出/余额减少都要填负数。
资金类数值（main_net_today_yi / main_net_5d_yi）统一换算成【亿元】；若图中是"万"，请除以10000换算成亿。"""


def call_gemini_chip(api_key, model, image_bytes, mime_type=None):
    """
    调用 Gemini 多模态分析筹码图。
    返回 (ok, result_dict, err_msg)。
    手机端兼容：直接用『原始字节 + MIME 类型』发给 Gemini，
    这样 iPhone 的 HEIC/HEIF、安卓的 WebP 等格式都能分析，不依赖本地 Pillow 解码。
    """
    if not _GENAI_OK:
        return False, None, "未安装 google-generativeai 库，请先 pip install。"
    if not api_key:
        return False, None, "未填写 Gemini API Key（请在左侧边栏填写）。"
    try:
        genai.configure(api_key=api_key)
        gmodel = genai.GenerativeModel(model or "gemini-1.5-flash")
        # MIME 兜底：拿不到或异常时按 jpeg 处理
        if not mime_type or "/" not in str(mime_type):
            mime_type = "image/jpeg"
        try:
            image_part = {"mime_type": mime_type, "data": image_bytes}
            resp = gmodel.generate_content(
                [CHIP_VISION_PROMPT, image_part],
                request_options={"timeout": max(HTTP_TIMEOUT, 90)},
            )
        except Exception:
            # 退路：普通 png/jpg 用 PIL 打开后再发
            if not _PIL_OK:
                raise
            img = Image.open(BytesIO(image_bytes))
            resp = gmodel.generate_content(
                [CHIP_VISION_PROMPT, img],
                request_options={"timeout": max(HTTP_TIMEOUT, 90)},
            )
        text = getattr(resp, "text", None) or ""
        data = _safe_parse_json(text)
        if data is None or "is_single_peak_low" not in data:
            return False, None, f"视觉模型返回无法解析：{text[:200]}"
        return True, data, ""
    except Exception as e:
        return False, None, f"调用 Gemini 失败：{e}"


def call_gemini_text(api_key, model, prompt_text, timeout=None):
    """
    通用 Gemini 纯文本调用（用于让 Gemini 也参与选股/买卖研判）。
    返回 (ok, text, err)。
    """
    if not _GENAI_OK:
        return False, None, "未安装 google-generativeai 库，请先 pip install。"
    if not api_key:
        return False, None, "未填写 Gemini API Key（请在左侧边栏填写）。"
    try:
        genai.configure(api_key=api_key)
        gmodel = genai.GenerativeModel(model or "gemini-1.5-flash")
        resp = gmodel.generate_content(
            prompt_text,
            request_options={"timeout": timeout or max(HTTP_TIMEOUT, 120)},
        )
        text = getattr(resp, "text", None) or ""
        return True, text, ""
    except Exception as e:
        return False, None, f"调用 Gemini 失败：{e}"


# ============================================================================
# 模块3.5：双模型综合研判（DeepSeek + Gemini 一起选股 / 判断买卖）
# ============================================================================

DUAL_SELECT_PROMPT = """你是A股硬科技产业链专家，精通『紫苏叶理论』。
紫苏叶公司标准（核心锚点业务需同时满足）：①产业链深层节点（Layer3+：底层硬件/核心材料/关键设备等"卖水人"）；②不可替代、壁垒高；③寡头垄断（有效竞争对手≤3家）。
不要拿公司"总盘子主业"一刀切否定，要主动挖掘其内部可能藏着的"紫苏叶锚点"细分业务。

""" + SERENITY_FRAMEWORK + """

""" + SERENITY_SCORE_GUIDE + """

""" + SERENITY_ANTIPATTERNS + """
命中伪概念排除条目越多，越要下调 is_perilla 与评分。
只返回一个JSON对象（不要任何多余文字、不要markdown标记）：
{"is_perilla": true或false, "anchor": "紫苏叶锚点业务名或null", "chain_layer": "卡在第几层（如：第4层 芯片/器件）", "antipattern_hits": "命中的伪概念排除条目（没有则填 无）", "reason": "大白话理由，点明卡在第几层、命中哪几条卡脖子特征，若命中伪概念也要说清，120字以内，让股票小白看懂", "factors": {"需求拐点":0-5,"架构耦合":0-5,"卡脖子严重度":0-5,"供应商集中度":0-5,"扩产难度":0-5,"证据质量":0-5,"估值偏离":0-5,"催化时机":0-5}, "penalties": {"增发摊薄":0-5,"公司治理":0-5,"炒作过热":0-5,"财务质量":0-5}}"""

DUAL_DECISION_PROMPT = """你是一位严格遵循『紫苏叶选股 + 戴维斯双击 + 右侧交易』的A股投资顾问。
我会给你一只股票的关键数据，以及系统规则引擎的初步结论。请你独立判断当前应采取的操作。
判断原则：
- 右侧交易：股价站上20日且30日均线才考虑买入；（尤其持仓时）跌破30日均线应卖出/离场。
- 不追高：获利盘过大、现价远高于平均成本、主力净流出、融资过热时，即使是好公司也应观望，别追。
- 戴维斯双击（净利润同比增长高 + PE处于近3年低分位）是重要加分买点。
- 数据缺失时要保守。
只返回一个JSON对象（不要任何多余文字、不要markdown标记）：
{"action": "买入" 或 "观望" 或 "卖出", "confidence": 0到1的小数, "reason": "大白话理由，120字以内，让股票小白看懂"}"""


def dual_select(name, deepseek_key, deepseek_model, gemini_key, gemini_model):
    """让 DeepSeek 与 Gemini 各自独立研判一家公司是否紫苏叶。返回 (ds, ds_err, gm, gm_err)。"""
    user_msg = f"请研判这家公司：{name}"
    # DeepSeek
    ds, ds_err = None, ""
    payload = {
        "model": deepseek_model or "deepseek-chat",
        "messages": [
            {"role": "system", "content": DUAL_SELECT_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    ok, content, err = _deepseek_post(payload, deepseek_key)
    if ok:
        ds = _safe_parse_json(content)
        if ds is None:
            ds_err = "DeepSeek 返回无法解析。"
    else:
        ds_err = err
    # Gemini
    gm, gm_err = None, ""
    ok2, text, err2 = call_gemini_text(
        gemini_key, gemini_model,
        DUAL_SELECT_PROMPT + "\n\n" + user_msg + "\n\n请只返回JSON。",
    )
    if ok2:
        gm = _safe_parse_json(text)
        if gm is None:
            gm_err = "Gemini 返回无法解析。"
    else:
        gm_err = err2
    return ds, ds_err, gm, gm_err


def _decision_facts(row, rule_sig, rule_reason, npr_threshold, pe_pct_threshold):
    """把一只股票的关键数据整理成给大模型看的文字清单。"""
    def g(k):
        v = row.get(k)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return MISSING
        return v
    lines = [
        f"股票：{row.get('name')}（{row.get('code')}）",
        f"是否持仓：{'是' if row.get('is_holding') else '否'}",
        f"是否符合紫苏叶（守门员判定）：{'是' if row.get('is_perilla_leaf') else '否'}"
        + ("（注意：人类强制收编，AI 守门员原本不认）" if row.get('is_override') else ""),
        f"收盘价：{g('close')}，MA10：{g('ma10')}，MA20：{g('ma20')}，MA30：{g('ma30')}",
        f"净利润同比增长：{g('npr_growth')}%（买入业绩门槛：> {npr_threshold}%）",
        f"PE：{g('pe')}，PE近3年分位：{g('pe_percentile')}%（买入估值门槛：< {pe_pct_threshold}%）",
        f"平均成本：{g('avg_cost')}，收盘获利比例：{g('profit_ratio')}%",
        f"龙虎榜净买入(万元)：{g('lhb_net')}，今日主力净流入(亿)：{g('main_net_today')}，"
        f"近5日主力净流入(亿)：{g('main_net_5d')}，融资余额变化：{g('margin_chg')}%",
        f"筹码：低位单峰密集={_chip_text(row, 'chip_single_peak')}，"
        f"站上平均成本线={_chip_text(row, 'chip_above_avg')}，高位发散={_chip_text(row, 'chip_high_diverge')}",
        f"系统规则引擎初步结论：{rule_sig} —— {rule_reason}",
    ]
    return "\n".join(str(x) for x in lines)


def dual_decision(row, deepseek_key, deepseek_model, gemini_key, gemini_model,
                  rule_sig, rule_reason, npr_threshold, pe_pct_threshold):
    """让 DeepSeek 与 Gemini 各自独立给出买/卖/观望判断。返回 (ds, ds_err, gm, gm_err)。"""
    facts = _decision_facts(row, rule_sig, rule_reason, npr_threshold, pe_pct_threshold)
    user_msg = facts + "\n\n请给出你的操作判断。"
    # DeepSeek
    ds, ds_err = None, ""
    payload = {
        "model": deepseek_model or "deepseek-chat",
        "messages": [
            {"role": "system", "content": DUAL_DECISION_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    ok, content, err = _deepseek_post(payload, deepseek_key)
    if ok:
        ds = _safe_parse_json(content)
        if ds is None:
            ds_err = "DeepSeek 返回无法解析。"
    else:
        ds_err = err
    # Gemini
    gm, gm_err = None, ""
    ok2, text, err2 = call_gemini_text(
        gemini_key, gemini_model,
        DUAL_DECISION_PROMPT + "\n\n" + user_msg + "\n\n请只返回JSON。",
    )
    if ok2:
        gm = _safe_parse_json(text)
        if gm is None:
            gm_err = "Gemini 返回无法解析。"
    else:
        gm_err = err2
    return ds, ds_err, gm, gm_err


# ============================================================================
# 模块3.6：紫苏叶评分卡（借鉴 serenity-skill 的加权打分，0~100 分）
# ============================================================================

# 8 个维度的权重（合计 100）
SERENITY_WEIGHTS = {
    "需求拐点": 15, "架构耦合": 10, "卡脖子严重度": 15, "供应商集中度": 12,
    "扩产难度": 12, "证据质量": 15, "估值偏离": 11, "催化时机": 10,
}
# 4 项风险扣分项（每项 0~5 分，扣分系数 ×2）
SERENITY_PENALTIES = ["增发摊薄", "公司治理", "炒作过热", "财务质量"]
PENALTY_FACTOR = 2.0


def _rating_0_5(v):
    """把 LLM 给的评分安全转成 0~5 的浮点；非法/缺失按 0 处理（容错，不崩）。"""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    if x < 0:
        return 0.0
    if x > 5:
        return 5.0
    return x


def serenity_score(factors, penalties=None):
    """
    紫苏叶评分卡：8 维度各 0~5 分按权重折算求和，再减去风险扣分（每分 ×2），clamp 到 0~100。
    缺失项按 0 分容错。返回 (score_int, detail_dict)。
    detail_dict 含各维度得分、原始分、扣分与档位，供 UI 折叠区展示明细。
    """
    factors = factors or {}
    penalties = penalties or {}
    detail = {"维度": {}, "扣分": {}}
    raw_total = 0.0
    for dim, weight in SERENITY_WEIGHTS.items():
        r = _rating_0_5(factors.get(dim))
        pts = round(r / 5.0 * weight, 1)
        raw_total += pts
        detail["维度"][dim] = {"原始分(0-5)": r, "权重": weight, "得分": pts}

    penalty_total = 0.0
    for p in SERENITY_PENALTIES:
        r = _rating_0_5(penalties.get(p))
        deduct = round(r * PENALTY_FACTOR, 1)
        penalty_total += deduct
        if r > 0:
            detail["扣分"][p] = {"风险分(0-5)": r, "扣分": deduct}

    score = raw_total - penalty_total
    score = max(0, min(100, int(round(score))))
    detail["原始合计"] = round(raw_total, 1)
    detail["扣分合计"] = round(penalty_total, 1)
    detail["最终评分"] = score
    detail["档位"] = serenity_grade(score)
    return score, detail


def serenity_grade(score):
    """把 0~100 评分翻译成大白话档位。score 为 None 时返回未评分。"""
    if score is None:
        return "未评分"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "未评分"
    if s >= 85:
        return "顶级优先"
    if s >= 70:
        return "高优先"
    if s >= 55:
        return "值得跟踪"
    return "早期/低优先"


def buyability_score(close, ma20, ma30):
    """
    【今日可买入度】只看"当天股价 vs 均线"判断现在是不是买点（与基本面紫苏叶评分相互独立）。
    返回 (buy_score:int 0~100 或 None, tag:大白话档位, why:一句话原因)。
    口径（右侧交易）：站上均线=可买、分高；还在均线下方=左侧寻底、暂别追、分低。
    """
    try:
        c = float(close)
        m30 = float(ma30)
    except (TypeError, ValueError):
        return None, "买点未知", "行情没拉到（股价/均线缺失），先刷新再看。"
    m20 = None
    try:
        m20 = float(ma20)
    except (TypeError, ValueError):
        m20 = None

    if m20 is not None and c > m20 and c > m30:
        return 90, "🟢 现在可买", "已站上20日和30日均价线，处于右侧上涨，可分批建仓。"
    if c > m30:
        return 60, "🟡 接近买点", "已站上30日均价线，但还没站上20日线，趋势待确认，可小仓试探或再等等。"
    return 30, "🔴 暂别追", "还在30日均价线下方，处于左侧寻底，先观望，等它站稳均线再说。"


def turnover_level(turnover):
    """
    【换手率情绪温度计】把换手率（%）映射到情绪档位。
    返回 (tag大白话, color色值, is_hot过热布尔, is_cold过冷布尔)。
    分级参考：<0.5%极冷 / 0.5-2%正常 / 2-5%活跃 / 5-10%偏热 / >10%极热。
    """
    if turnover is None:
        return "未知", "#888888", False, False
    try:
        t = float(turnover)
    except (TypeError, ValueError):
        return "未知", "#888888", False, False
    if t < 0.5:
        return "🔵 极冷场", "#4a90d9", False, True
    if t < 2.0:
        return "⚪ 正常换手", "#888888", False, False
    if t < 5.0:
        return "🟡 活跃换手", "#e6b800", False, False
    if t < 10.0:
        return "🟠 偏热注意", "#e67e22", True, False
    return "🔴 极热警惕", "#c0392b", True, False


def eval_buyability_for_sectors(sectors):
    """为挖掘结果里每只股票评估『今日可买入度』（联网取价算分），结果写回 stk['_buy_*']。单只失败不影响其他。"""
    for sec in sectors or []:
        for stk in sec.get("stocks", []) or []:
            code = str(stk.get("code") or "")
            if not code:
                stk["_buy_score"], stk["_buy_tag"], stk["_buy_why"] = None, "买点未知", "缺少股票代码"
                continue
            try:
                ma = fetch_price_ma(code)
                bs, tag, why = buyability_score(ma.get("close"), ma.get("ma20"), ma.get("ma30"))
                stk["_buy_score"], stk["_buy_tag"], stk["_buy_why"] = bs, tag, why
                stk["_close"], stk["_ma20"], stk["_ma30"] = ma.get("close"), ma.get("ma20"), ma.get("ma30")
            except Exception:
                stk["_buy_score"], stk["_buy_tag"], stk["_buy_why"] = None, "买点未知", "行情没拉到（可重试）"
    return sectors


def _clamp_score(v):
    """把分数夹到 0~100 的整数。"""
    try:
        return int(max(0, min(100, round(v))))
    except (TypeError, ValueError):
        return 0


def next_day_buy_score(stk, turnover, vol_ratio, rally, rally_threshold):
    """
    【紫苏叶当日精选·次日买入推荐度】对一只挖掘候选股按"当天股价 + 量价 + 涨幅"打分。
    返回 (score:int 0~100, reasons:list[str], exclude:bool, exclude_reason:str)。
    先判暴涨排除（涨幅超滑块阈值 或 高出30日线>30%），再综合打分。
    """
    close = stk.get("_close")
    ma20 = stk.get("_ma20")
    ma30 = stk.get("_ma30")
    serenity = stk.get("_score")
    gain_pct = (rally or {}).get("gain_pct")

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    c, m20, m30 = _f(close), _f(ma20), _f(ma30)

    # —— 暴涨排除（先判，避免追高）——
    ext30 = None
    if c is not None and m30 is not None and m30 > 0:
        ext30 = (c - m30) / m30 * 100
    if gain_pct is not None and gain_pct > rally_threshold:
        return 0, [], True, f"近2-3月已涨 {gain_pct}%（超过你设的 {rally_threshold}% 红线），追高风险大，先不推荐。"
    if ext30 is not None and ext30 > 30:
        return 0, [], True, f"股价已高出30日均价线 {round(ext30,1)}%（涨过头了），追高风险大，先不推荐。"

    # —— 综合打分 ——
    score = 0
    if m20 is not None and c is not None and m30 is not None and c > m20 and c > m30:
        score += 60   # 站上双线（右侧上涨）
    elif c is not None and m30 is not None and c > m30:
        score += 40   # 仅站上30日线
    else:
        score += 15   # 还在均线下

    # 紫苏叶卡位加分（基本面好坏，最多 +20）
    score += round((_f(serenity) or 0) / 100 * 20)

    # 量价
    vr = _f(vol_ratio)
    if vr is not None and vr > 1.5 and c is not None and m20 is not None and c > m20:
        score += 12   # 放量突破
    elif vr is not None and 1.0 <= vr <= 1.5:
        score += 6    # 温和放量
    elif vr is not None and vr < 0.7 and c is not None and m20 is not None and m20 > 0 and abs(c - m20) / m20 < 0.02:
        score += 8    # 缩量回踩贴均线

    # 换手率
    to = _f(turnover)
    lv_tag, _, is_hot, _is_cold = turnover_level(to)
    if to is not None and 2.0 <= to <= 5.0:
        score += 5    # 活跃换手
    if is_hot:
        score -= 10   # 过热扣分

    # 不过度延伸（贴着均线上方更稳）
    if ext30 is not None and 0 <= ext30 <= 15:
        score += 8

    score = _clamp_score(score)

    # —— 大白话买入理由（复用多空对比的做多清单）——
    synth = {
        "close": close, "ma20": ma20, "ma30": ma30,
        "serenity_score": serenity, "vol_ratio": vol_ratio, "turnover": turnover,
    }
    bulls, _bears = _bull_bear_signals(synth)
    reasons = list(bulls)
    if gain_pct is not None:
        reasons.append(f"近2-3月涨幅 {gain_pct}%，不算过热，没追高风险")
    if not reasons:
        reasons.append("技术面中规中矩，可小仓试探")
    return score, reasons, False, ""


def tech_buy_score(close, ma10, ma20, ma30, turnover, vol_ratio, rally, rally_threshold):
    """
    【热门板块龙头·技术买点】纯技术线打分（不要求紫苏叶卡位），用于热门板块成分股。
    返回 (score:int 0~100, reasons:list[str], caution:str, exclude:bool, exclude_reason:str)。
    暴涨排除口径与 next_day_buy_score 完全一致（同一滑块阈值）。
    """
    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    c, m20, m30 = _f(close), _f(ma20), _f(ma30)
    gain_pct = (rally or {}).get("gain_pct")

    # —— 暴涨排除（先判）——
    ext30 = None
    if c is not None and m30 is not None and m30 > 0:
        ext30 = (c - m30) / m30 * 100
    if gain_pct is not None and gain_pct > rally_threshold:
        return 0, [], "", True, f"该板块虽热，但这只近2-3月已涨 {gain_pct}%（超 {rally_threshold}% 红线），追高风险大，先不推荐。"
    if ext30 is not None and ext30 > 30:
        return 0, [], "", True, f"该板块虽热，但这只已高出30日线 {round(ext30,1)}%（涨过头），追高风险大，先不推荐。"

    # —— 技术打分（不含紫苏叶）——
    score = 0
    if m20 is not None and c is not None and m30 is not None and c > m20 and c > m30:
        score += 60
    elif c is not None and m30 is not None and c > m30:
        score += 40
    else:
        score += 20

    vr = _f(vol_ratio)
    if vr is not None and vr > 1.5 and c is not None and m20 is not None and c > m20:
        score += 12
    elif vr is not None and 1.0 <= vr <= 1.5:
        score += 6
    elif vr is not None and vr < 0.7 and c is not None and m20 is not None and m20 > 0 and abs(c - m20) / m20 < 0.02:
        score += 8

    to = _f(turnover)
    lv_tag, _, is_hot, _is_cold = turnover_level(to)
    if to is not None and 2.0 <= to <= 5.0:
        score += 5

    caution_parts = []
    if ext30 is not None and ext30 > 15:
        score -= 8
        caution_parts.append(f"已高出30日线 {round(ext30,1)}%，有点偏高，别追太猛")
    if is_hot:
        score -= 10
        caution_parts.append(f"换手率偏热（{to}%，{lv_tag}），情绪亢奋需留意")

    score = _clamp_score(score)

    synth = {
        "close": close, "ma20": ma20, "ma30": ma30,
        "vol_ratio": vol_ratio, "turnover": turnover,
    }
    bulls, _bears = _bull_bear_signals(synth)
    reasons = list(bulls)
    if gain_pct is not None:
        reasons.append(f"近2-3月涨幅 {gain_pct}%，未过热")
    if not reasons:
        reasons.append("技术面中规中矩，可小仓试探")
    caution = "；".join(caution_parts)
    return score, reasons, caution, False, ""


def eval_daily_picks(sectors, rally_threshold):
    """
    【A·紫苏叶当日精选】从挖掘候选里挑次日买入 Top5。
    返回 (picks:list[dict], excluded_cnt:int)。pick={code,name,sector,score,reasons,gain_pct}。
    """
    eval_buyability_for_sectors(sectors)  # 先保证有价/均线
    seen = set()
    candidates = []
    for sec in sectors or []:
        sec_name = sec.get("sector") or sec.get("name") or "未命名赛道"
        for stk in sec.get("stocks", []) or []:
            code = str(stk.get("code") or "")
            if not code or code in seen:
                continue
            seen.add(code)
            candidates.append((sec_name, stk))

    picks = []
    excluded_cnt = 0
    for sec_name, stk in candidates:
        code = str(stk.get("code") or "")
        try:
            turnover, vol_ratio = fetch_turnover_vol(code)
            rally = fetch_recent_rally(code)
            score, reasons, exclude, exclude_reason = next_day_buy_score(
                stk, turnover, vol_ratio, rally, rally_threshold
            )
            if exclude:
                excluded_cnt += 1
                continue
            picks.append({
                "code": code,
                "name": stk.get("name") or code,
                "sector": sec_name,
                "score": score,
                "serenity": stk.get("_score"),
                "reasons": reasons,
                "gain_pct": (rally or {}).get("gain_pct"),
            })
        except Exception:
            continue
    picks.sort(key=lambda p: -(p.get("score") or 0))
    return picks[:5], excluded_cnt


def eval_hot_board_picks(top_boards=3, per_board=3, rally_threshold=60):
    """
    【B·热门板块龙头·技术买点】取最热概念板块，每板块挑 per_board 只技术买点股（排除暴涨）。
    返回 [{board, pct, picks:[{code,name,score,reasons,caution,gain_pct}], excluded}]。
    单只/单板块失败跳过，整体不崩。
    返回 (result:list, err:str)。err 非空表示连热门板块列表都没取到（含诊断信息）。
    """
    boards, err = fetch_hot_boards(top_boards)
    if not boards:
        return [], err
    result = []
    for b in boards or []:
        board_name = b.get("board")
        board_pct = b.get("pct")
        if not board_name:
            continue
        try:
            cons = fetch_board_cons(board_name)
        except Exception:
            cons = None
        if cons is None or len(cons) == 0:
            continue

        # —— 用成分表里的廉价字段预筛 shortlist（避免对全板块联网）——
        try:
            df = cons.copy()
            if "pct" in df.columns:
                df["pct"] = pd.to_numeric(df["pct"], errors="coerce")
                # 今日涨幅在 -3%~7%（非涨停、非大跌），更可能是健康买点
                df = df[(df["pct"] >= -3) & (df["pct"] <= 7)]
                df = df.sort_values("pct", ascending=False)
            shortlist = df.head(8)
        except Exception:
            shortlist = cons.head(8)

        picks = []
        excluded = 0
        for _i, r in shortlist.iterrows():
            code = str(r.get("code") or "").zfill(6)
            name = r.get("name") or code
            if not code or code == "000000":
                continue
            try:
                ma = fetch_price_ma(code)
                turnover, vol_ratio = fetch_turnover_vol(code)
                rally = fetch_recent_rally(code)
                score, reasons, caution, exclude, _exr = tech_buy_score(
                    ma.get("close"), ma.get("ma10"), ma.get("ma20"), ma.get("ma30"),
                    turnover, vol_ratio, rally, rally_threshold
                )
                if exclude:
                    excluded += 1
                    continue
                picks.append({
                    "code": code, "name": name, "score": score,
                    "reasons": reasons, "caution": caution,
                    "gain_pct": (rally or {}).get("gain_pct"),
                })
            except Exception:
                continue
        picks.sort(key=lambda p: -(p.get("score") or 0))
        result.append({
            "board": board_name,
            "pct": board_pct,
            "picks": picks[:per_board],
            "excluded": excluded,
        })
    return result, ""


def _avg_factor_dicts(*dicts):
    """把多个 {维度:分} 字典按维度求平均（用于双AI评分合并）。空输入返回 {}。"""
    valid = [d for d in dicts if isinstance(d, dict) and d]
    if not valid:
        return {}
    keys = set()
    for d in valid:
        keys.update(d.keys())
    out = {}
    for k in keys:
        vals = [_rating_0_5(d.get(k)) for d in valid if k in d]
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def compute_and_save_score(code, factors, penalties=None):
    """算出紫苏叶评分并写库；factors 为空则不写。返回 (score, detail) 或 (None, None)。"""
    if not factors:
        return None, None
    score, detail = serenity_score(factors, penalties)
    try:
        update_fields(code, {
            "serenity_score": score,
            "score_detail": json.dumps(detail, ensure_ascii=False),
        })
    except Exception:
        pass
    return score, detail


# ============================================================================
# 模块4：终极买卖决策引擎
# ============================================================================

def is_davis_double(row, npr_threshold, pe_pct_threshold):
    """戴维斯双击：净利润同比增长 > 阈值 且 PE 分位 < 阈值。数据缺失则视为不满足。"""
    npr = row.get("npr_growth")
    pe_pct = row.get("pe_percentile")
    if npr is None or pe_pct is None:
        return False
    try:
        return float(npr) > npr_threshold and float(pe_pct) < pe_pct_threshold
    except Exception:
        return False


def decide(row, npr_threshold=20.0, pe_pct_threshold=50.0):
    """
    核心决策引擎。输入一行股票数据（dict），返回 (signal_key, reason_大白话)。
    优先级短路：持仓破位风控 > 视觉必需gate > 紫苏叶买入/观望逻辑。
    """
    close = row.get("close")
    ma20 = row.get("ma20")
    ma30 = row.get("ma30")
    is_holding = bool(row.get("is_holding"))
    is_perilla = bool(row.get("is_perilla_leaf"))
    has_chip = bool(row.get("has_chip"))

    high_diverge = bool(row.get("chip_high_diverge")) if has_chip else False
    single_peak = bool(row.get("chip_single_peak")) if has_chip else False

    # 行情数据缺失，无法判断（注意：这跟筹码图无关，是股价/均线没拉到）
    if close is None or ma30 is None:
        return "待刷新行情数据", "股价/均线数据还没拉到（可能是刚才网络抽风）。请点左侧『🔄 一键刷新全池数据』再试一次，通常重试就好。"

    # 计算"高位追高风险"清单（用筹码获利比例 / 平均成本 / 龙虎榜净额）
    risks = _high_position_risks(row, close)
    high_risk = len(risks) > 0

    # 1) 持仓者破位风控（最高优先）
    if is_holding and (close < ma30 or high_diverge):
        if close < ma30:
            return "坚决清仓卖出", f"你持有的这只股，股价（{close}）已跌破30日均价线（{ma30}）这条重要生命线，按纪律应坚决离场止损。"
        return "坚决清仓卖出", "你持有的这只股出现筹码『高位发散』，主力可能在高位派发，建议坚决离场。"

    # 2) 筹码数据缺失：不再"无图就拦死"，而是基于股价/均线照常给建议，仅附一句温馨提示。
    #    （系统会自动抓取筹码；若自动也没抓到、又没传图，则筹码相关加分/风控项不参与。）
    chip_hint = "" if has_chip else "（温馨提示：这只股的筹码数据暂时没拿到，下面结论主要看股价和均线；想要更精准的买卖点，可到『上传筹码图』页面补一张图。）"

    # 3) 紫苏叶逻辑成立
    if is_perilla:
        if ma20 is not None and close > ma20 and close > ma30:
            # 站上均线，处于多头
            if is_holding:
                tip = f"移动止盈提示：跌破10日均价线（{row.get('ma10')}）可考虑减仓，跌破30日均价线（{ma30}）则清仓。"
                if high_risk:
                    return "持有移动止盈", "趋势仍在均线之上，可继续持有，但要警惕高位风险：" + "；".join(risks) + "。" + tip + chip_hint
                return "持有移动止盈", f"股价（{close}）稳稳站在均价线之上，趋势健康，继续持有。{tip}{chip_hint}"
            else:
                # 未持仓：高位则不建议追高买入（安全保护）
                if high_risk:
                    return "只看不动观望", "是符合紫苏叶标准的好公司，技术上也站上了均线，但【现在是高位，不建议追高买入】：" + "；".join(risks) + "。建议等股价回调到均线附近、获利盘消化后再考虑。" + chip_hint
                davis = is_davis_double(row, npr_threshold, pe_pct_threshold)
                # 卡位优先级：紫苏叶评分≥70（卡得很死的上游环节）也作为强烈买入的加分触发
                sc = row.get("serenity_score")
                strong_score = False
                try:
                    strong_score = sc is not None and float(sc) >= 70
                except (TypeError, ValueError):
                    strong_score = False
                # 量价确认：量比>2且收盘站上均线（放量突破，买盘积极）
                vr = None
                try:
                    vr = float(row.get("vol_ratio")) if row.get("vol_ratio") is not None else None
                except (TypeError, ValueError):
                    vr = None
                vol_breakthrough = bool(vr is not None and vr > 2.0)
                if single_peak or davis or strong_score or vol_breakthrough:
                    why = []
                    if single_peak:
                        why.append("筹码处于低位单峰密集（成本集中、抛压小）")
                    if davis:
                        why.append("业绩大涨且估值偏低（戴维斯双击）")
                    if strong_score:
                        try:
                            _sc_str = f"{int(float(sc))} 分"
                        except (TypeError, ValueError):
                            _sc_str = "高分"
                        why.append(f"产业链卡位极硬（紫苏叶评分 {_sc_str}，{serenity_grade(sc)}）")
                    if vol_breakthrough:
                        why.append(f"放量突破均线（量比{vr}倍），大单积极买入")
                    return "强烈买入", "符合紫苏叶好公司，且股价站上均线，又叠加" + "、".join(why) + "，是难得的好买点，可分批建仓。" + chip_hint
                return "分批建仓买入", f"这是符合紫苏叶标准的好公司，股价（{close}）已站上20日和30日均价线，进入右侧上涨，可分批建仓买入。{chip_hint}"
        elif close < ma30:
            return "只看不动观望", f"好公司，但股价（{close}）还在30日均价线（{ma30}）下方，处于左侧寻底阶段，先观望，等它站稳均线再说。{chip_hint}"
        else:
            return "只看不动观望", f"好公司，股价（{close}）在30日均价线之上但还没站上20日线（{ma20}），趋势未完全走强，先观望。{chip_hint}"

    # 兜底
    return "只看不动观望", "暂不满足明确的买入或卖出条件，保持观望。"


def _high_position_risks(row, close):
    """根据 获利比例 / 平均成本 / 龙虎榜净额 / 主力资金流 / 融资变化，列出"高位追高"风险点（大白话）。"""
    risks = []
    profit = row.get("profit_ratio")
    avg_cost = row.get("avg_cost")
    lhb_net = row.get("lhb_net")
    main_5d = row.get("main_net_5d")
    margin_chg = row.get("margin_chg")
    try:
        if profit is not None and float(profit) >= 90:
            risks.append(f"几乎全员获利（获利盘 {profit}%），随时可能有人获利了结")
    except Exception:
        pass
    try:
        if avg_cost and close and float(close) > float(avg_cost) * 1.20:
            risks.append(f"现价（{close}）比市场平均成本（{avg_cost}）高出两成以上，追高成本偏贵")
    except Exception:
        pass
    try:
        if lhb_net is not None and float(lhb_net) < 0:
            risks.append(f"龙虎榜主力净卖出（{lhb_net} 万元），资金在高位出货")
    except Exception:
        pass
    try:
        if main_5d is not None and float(main_5d) < 0:
            risks.append(f"近5日主力资金净流出（{main_5d} 亿元），中线大资金在撤退")
    except Exception:
        pass
    try:
        if margin_chg is not None and float(margin_chg) >= 5:
            risks.append(f"融资余额猛增（{margin_chg}%），杠杆资金追高、情绪偏过热")
    except Exception:
        pass
    return risks


def _bull_bear_signals(row):
    """汇总一只股的『做多理由』与『做空理由』两份大白话清单，供多空对比卡展示。"""
    bulls, bears = [], []
    close = row.get("close")
    ma20 = row.get("ma20")
    ma30 = row.get("ma30")

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    c, m20, m30 = _f(close), _f(ma20), _f(ma30)

    # —— 趋势/均线 ——
    if c is not None and m30 is not None:
        if m20 is not None and c > m20 and c > m30:
            bulls.append(f"股价（{close}）站上20日和30日均价线，处于右侧上涨")
        elif c > m30:
            bulls.append(f"股价（{close}）站在30日均价线（生命线）之上")
        if c < m30:
            bears.append(f"股价（{close}）跌破30日均价线（{ma30}）这条重要生命线")
        elif m20 is not None and c < m20:
            bears.append(f"股价（{close}）还没站上20日均价线（{ma20}），趋势偏弱")

    # —— 筹码 ——
    if row.get("has_chip"):
        if row.get("chip_single_peak"):
            bulls.append("筹码低位单峰密集（成本集中、抛压小）")
        if row.get("chip_above_avg"):
            bulls.append("股价站上市场平均成本线（多数持有人盈利）")
        if row.get("chip_high_diverge"):
            bears.append("筹码高位发散（获利盘大、主力可能在派发）")

    # —— 业绩 + 估值（戴维斯双击，用默认阈值 20% / 50% 粗判）——
    npr = _f(row.get("npr_growth"))
    pe_pct = _f(row.get("pe_percentile"))
    if npr is not None and npr > 20:
        bulls.append(f"净利润同比大涨（{npr}%），业绩向好")
    if npr is not None and npr < 0:
        bears.append(f"净利润同比下滑（{npr}%），业绩承压")
    if pe_pct is not None and pe_pct < 50:
        bulls.append(f"估值处于近3年偏低位置（PE分位{pe_pct}%），不算贵")
    if pe_pct is not None and pe_pct >= 80:
        bears.append(f"估值处于近3年高位（PE分位{pe_pct}%），偏贵")

    # —— 紫苏叶卡位 ——
    sc = _f(row.get("serenity_score"))
    if sc is not None and sc >= 70:
        bears_note = ""
        bulls.append(f"产业链卡位极硬（紫苏叶评分 {int(sc)} 分，{serenity_grade(sc)}）{bears_note}")

    # —— 股东户数 ——
    chg = _f(row.get("gdhs_chg"))
    if chg is not None:
        if chg < 0:
            bulls.append(f"股东户数较上期减少 {abs(chg)}%（筹码在集中，常是利好）")
        elif chg > 0:
            bears.append(f"股东户数较上期增加 {chg}%（筹码在分散，需留意）")

    # —— 量价信号（换手率情绪 + 量比）——
    vr = _f(row.get("vol_ratio"))
    to = _f(row.get("turnover"))
    is_holding = bool(row.get("is_holding"))
    lv_tag, _, is_hot, is_cold = turnover_level(to)
    # 放量突破：量比>2且收盘高于20日线
    if vr is not None and vr > 2.0 and c is not None and m20 is not None and c > m20:
        bulls.append(f"放量突破（量比{vr}倍），大单积极买入，动能较强")
    # 缩量回踩：量比<0.7且价格贴近MA20（误差2%以内）
    if vr is not None and vr < 0.7 and c is not None and m20 is not None and abs(c - m20) / m20 < 0.02:
        bulls.append(f"缩量回踩20日均线（量比{vr}倍），抛压小、蓄势待发，是不错的入场机会")
    # 放量下跌警告：量比>2且收盘低于均线
    if vr is not None and vr > 2.0 and c is not None and m20 is not None and c < m20:
        bears.append(f"放量下跌（量比{vr}倍），卖盘沉重，需警惕持续下行")
    # 换手率情绪
    if to is not None and is_hot:
        bears.append(f"换手率过热（{to}%，{lv_tag}），市场过于亢奋，顶部风险加大")
    if to is not None and is_cold and not is_holding:
        bulls.append(f"换手极冷（{to}%，{lv_tag}），低温蓄势往往是底部信号之一")

    # —— 高位追高风险（复用现成清单）——
    for r in _high_position_risks(row, close):
        bears.append(r)

    return bulls, bears


def render_bull_bear(row):
    """在卡片下方渲染『做多理由 vs 做空理由』双栏对比（深色 HTML 卡片样式，参考图1）。"""
    bulls, bears = _bull_bear_signals(row)

    def _items_html(items, empty_txt):
        if not items:
            return f'<div style="color:#888;font-size:13px;padding:4px 0;">{empty_txt}</div>'
        rows_html = "".join(
            f'<div style="margin:5px 0;font-size:14px;line-height:1.5;">'
            f'<span style="margin-right:6px;">→</span>{item}</div>'
            for item in items
        )
        return rows_html

    bull_html = _items_html(bulls, "暂无明显看涨信号")
    bear_html = _items_html(bears, "暂无明显看跌信号")

    st.markdown(
        f"""
        <div style="display:flex;gap:12px;margin:10px 0 4px 0;">
          <div style="flex:1;background:#1a2e1a;border-left:4px solid #4caf50;
                      border-radius:10px;padding:14px 16px;min-height:80px;">
            <div style="color:#6fcf7f;font-weight:700;font-size:15px;margin-bottom:10px;">
              📈 多方逻辑
            </div>
            <div style="color:#c8e6c9;">{bull_html}</div>
          </div>
          <div style="flex:1;background:#2e1a1a;border-left:4px solid #e05252;
                      border-radius:10px;padding:14px 16px;min-height:80px;">
            <div style="color:#e07070;font-weight:700;font-size:15px;margin-bottom:10px;">
              📉 空方风险
            </div>
            <div style="color:#ffcdd2;">{bear_html}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_bars(score_detail_json):
    """
    【紫苏叶评分维度进度条】把 score_detail JSON 里每个维度渲染成横向进度条（参考图2风格）。
    rating ≥4 → 红色；≥3 → 橙色；其他 → 灰色。容错：解析失败直接 return。
    """
    if not score_detail_json:
        return
    try:
        detail = json.loads(score_detail_json)
    except Exception:
        return
    dims = detail.get("维度", {})
    if not dims:
        return

    bars_html = '<div style="margin:10px 0;">'
    bars_html += '<div style="color:#aaa;font-size:13px;margin-bottom:8px;">🌿 紫苏叶评分分项（0-5分）</div>'
    for name, info in dims.items():
        try:
            rating = float(info.get("原始分(0-5)") or info.get("rating") or 0)
        except (TypeError, ValueError):
            rating = 0
        pct = min(100, int(rating / 5 * 100))
        if rating >= 4:
            color = "#e05252"
        elif rating >= 3:
            color = "#e6823c"
        elif rating >= 2:
            color = "#d4ac0d"
        else:
            color = "#555555"
        bars_html += (
            f'<div style="display:flex;align-items:center;margin:5px 0;">'
            f'<span style="display:inline-block;width:110px;color:#bbb;font-size:13px;'
            f'flex-shrink:0;">{name}</span>'
            f'<div style="flex:1;height:8px;background:#2a2a2a;border-radius:4px;'
            f'margin:0 10px;max-width:200px;">'
            f'<div style="width:{pct}%;height:100%;background:{color};border-radius:4px;"></div>'
            f'</div>'
            f'<span style="color:#bbb;font-size:12px;min-width:28px;">{rating:.0f}/5</span>'
            f'</div>'
        )

    # 风险扣分项
    deductions = detail.get("扣分", {})
    if deductions:
        ded_items = []
        for k, v in deductions.items():
            try:
                d = v.get("扣分") or v.get("deduction") or 0
                if d:
                    ded_items.append(f"{k} -{d}")
            except Exception:
                pass
        if ded_items:
            bars_html += (
                f'<div style="color:#e07070;font-size:12px;margin-top:8px;">'
                f'⚠ 风险扣分：{"；".join(ded_items)}</div>'
            )

    bars_html += '</div>'
    st.markdown(bars_html, unsafe_allow_html=True)


# ============================================================================
# 模块5：Streamlit 小白友好 UI
# ============================================================================

def render_signal_card(row, signal_key, reason):
    """渲染一张大色块操作建议卡片。"""
    style = SIGNAL_STYLE.get(signal_key, SIGNAL_STYLE["只看不动观望"])
    name = row.get("name") or ""
    code = row.get("code") or ""
    hold_tag = "（已持仓）" if row.get("is_holding") else "（未持仓）"
    override_tag = " 👑人类强制收编" if row.get("is_override") else ""
    # 紫苏叶评分徽章（卡位有多硬），无评分时给提示
    _sc = row.get("serenity_score")
    try:
        _sc_int = int(float(_sc))  # NaN / None 均会抛异常，统一进 except
        _sc_valid = True
    except (TypeError, ValueError):
        _sc_valid = False
    if _sc_valid:
        score_badge = (f'<span style="background:rgba(255,255,255,0.25);border-radius:8px;'
                       f'padding:2px 10px;font-size:15px;font-weight:700;margin-left:8px;">'
                       f'🌿 紫苏叶评分 {_sc_int}/100 · {serenity_grade(_sc)}</span>')
    else:
        score_badge = ('<span style="background:rgba(255,255,255,0.18);border-radius:8px;'
                       'padding:2px 10px;font-size:14px;margin-left:8px;">🌿 未评分（重新研判可生成）</span>')
    st.markdown(
        f"""
        <div style="background:{style['color']};color:{style['text']};
                    padding:18px 20px;border-radius:14px;margin-bottom:14px;">
          <div class="perilla-card-title" style="font-size:22px;font-weight:800;">
            {style['emoji']} {name} {code} {hold_tag}{override_tag} —— {signal_key}
          </div>
          <div style="margin-top:6px;">{score_badge}</div>
          <div class="perilla-card-reason" style="font-size:16px;margin-top:8px;line-height:1.6;">{reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt(v, suffix=""):
    """格式化展示：None -> 数据缺失。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return MISSING
    return f"{v}{suffix}"


def check_login():
    """
    登录闸门：未登录就显示登录框并 st.stop()（后面的内容都不会运行）；
    登录成功返回用户名。
    账号密码存放在 Streamlit 的 Secrets（st.secrets["passwords"]）里，
    格式为 [passwords] 区段下『用户名 = "密码"』，这样密码不会出现在公开代码里。
    """
    # 已登录 → 直接放行
    if st.session_state.get("auth_ok"):
        return st.session_state.get("auth_user")

    # 读取凭据库（来自 Secrets）
    try:
        creds = dict(st.secrets["passwords"])
    except Exception:
        creds = {}

    st.title("🌿 紫苏叶 AI 投研系统 · 登录")

    # 还没配置任何账号 → 给出小白可照做的指引
    if not creds:
        st.error("⚠️ 系统还没有配置任何登录账号，暂时无法使用。")
        st.markdown(
            "**管理员设置方法（一次性）：**\n\n"
            "- 在线版（Streamlit Cloud）：进入 App 的 **Settings → Secrets**，粘贴下面这段，"
            "把用户名/密码改成你要的：\n\n"
            "```toml\n[passwords]\n小明 = \"my-password-123\"\n小红 = \"another-pass-456\"\n```\n\n"
            "- 本地电脑测试：在项目里新建文件 `.streamlit/secrets.toml`，写入同样的内容。\n\n"
            "保存后刷新本页即可登录。每个账号登录后只能看到自己的股票池和自己填的 API Key。"
        )
        st.stop()

    with st.form("login_form"):
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录", use_container_width=True, type="primary")

    if submitted:
        real = creds.get(username)
        # 用 hmac.compare_digest 做恒定时间比较，避免泄露密码长度等信息
        if real is not None and hmac.compare_digest(str(real), str(password)):
            st.session_state["auth_ok"] = True
            st.session_state["auth_user"] = username
            st.rerun()
        else:
            st.error("用户名或密码不对，请重试。")

    st.caption("🔒 这是私人系统，登录后每个人只能看到自己的股票池和自己填写的 API Key，互不可见。")
    st.stop()


def main():
    st.set_page_config(
        page_title="紫苏叶 AI 投研系统",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="collapsed",  # 手机上默认收起侧边栏，先看正文
    )

    # 登录闸门：未登录会在此显示登录框并停下，下面的代码都不会执行
    auth_user = check_login()

    init_db()  # 注意：此时已登录，建/连的是该用户专属的数据库

    # ---------------- 手机端友好的响应式样式 ----------------
    st.markdown(
        """
        <style>
        /* 通用：内容区留白收紧一点，手机上不浪费空间 */
        .block-container { padding-top: 1.2rem; padding-bottom: 3rem; }

        /* 按钮更大更好按（触摸友好），文字不换行挤压 */
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
            min-height: 44px;
            border-radius: 10px;
            font-size: 16px;
        }

        /* 标签页可横向滑动，5个tab在手机上不被压扁 */
        div[data-baseweb="tab-list"] { overflow-x: auto; flex-wrap: nowrap; }
        button[data-baseweb="tab"] { white-space: nowrap; }

        /* 表格在窄屏可左右滑动查看 */
        div[data-testid="stDataFrame"] { overflow-x: auto; }

        /* ====== 手机屏幕（宽度 <= 640px）专属优化 ====== */
        @media (max-width: 640px) {
            /* 内容贴边一点，争取更多可视宽度；顶部多留点空，标题不顶到工具栏 */
            .block-container { padding-left: 0.8rem; padding-right: 0.8rem; padding-top: 1.6rem; }

            /* 标题缩小，避免占满整屏；line-height 放宽 + 留上边距，防止中文字顶部被裁切 */
            h1 {
                font-size: 1.4rem !important;
                line-height: 1.55 !important;
                padding-top: 0.4rem !important;
                margin-top: 0.2rem !important;
                overflow: visible !important;
            }
            h2 { font-size: 1.2rem !important; line-height: 1.5 !important; }
            h3 { font-size: 1.05rem !important; line-height: 1.5 !important; }

            /* 关键：让并排的列在手机上自动竖向堆叠，不再左右挤成一团 */
            div[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 0.4rem !important; }
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                flex: 1 1 100% !important;
                width: 100% !important;
                min-width: 100% !important;
            }

            /* 所有按钮在手机上占满整行，方便单手点 */
            .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
                width: 100% !important;
            }

            /* 大色块操作卡片：标题字号略缩，照样醒目 */
            .perilla-card-title { font-size: 18px !important; }
            .perilla-card-reason { font-size: 15px !important; }

            /* 输入框/下拉字号 16px，避免 iOS Safari 自动放大页面 */
            input, textarea, select { font-size: 16px !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🌿 硬科技【紫苏叶 AI 投研与综合择时系统】")
    st.caption("找到产业链最底层、别人离不开的好公司，在合适的时机告诉你该买、该卖还是该等。"
               "📱 手机用户：点左上角 » 可展开『设置/API Key』。")

    # ---------------- 侧边栏 ----------------
    with st.sidebar:
        # 当前登录用户 + 退出登录
        st.success(f"👤 当前用户：**{auth_user}**")
        if st.button("🚪 退出登录", use_container_width=True):
            for k in ("auth_ok", "auth_user"):
                st.session_state.pop(k, None)
            st.rerun()
        st.divider()

        st.header("⚙️ 设置")
        st.subheader("AI 钥匙（API Key）")

        # 读取该用户上次"记住"的钥匙作为默认值，实现登录后自动填好、无需重输
        saved_ds_key = load_setting("deepseek_key", "")
        saved_ds_model = load_setting("deepseek_model", "deepseek-chat")
        saved_gm_key = load_setting("gemini_key", "")
        saved_gm_model = load_setting("gemini_model", "gemini-1.5-flash")
        has_saved_keys = bool(saved_ds_key or saved_gm_key)

        deepseek_key = st.text_input("DeepSeek API Key", value=saved_ds_key, type="password",
                                     help="用于『AI 选股研判』。从 deepseek.com 申请。")
        deepseek_model = st.text_input("DeepSeek 模型名", value=saved_ds_model,
                                       help="一般保持默认即可。")
        gemini_key = st.text_input("Gemini API Key", value=saved_gm_key, type="password",
                                   help="用于『筹码分布图』看图分析。从 Google AI Studio 申请。")
        gemini_model = st.text_input("Gemini 模型名", value=saved_gm_model,
                                     help="一般保持默认即可。")

        remember_keys = st.checkbox("💾 记住我的钥匙（下次登录自动填好）", value=has_saved_keys,
                                    help="勾选并点下方按钮后，钥匙会保存在你专属的本地数据库里，下次登录自动填好，无需重输。")
        if st.button("保存钥匙设置", use_container_width=True):
            if remember_keys:
                save_setting("deepseek_key", deepseek_key)
                save_setting("deepseek_model", deepseek_model)
                save_setting("gemini_key", gemini_key)
                save_setting("gemini_model", gemini_model)
                st.success("已记住，下次登录会自动填好。")
            else:
                # 取消记住 → 清空已保存的钥匙
                for _k in ("deepseek_key", "gemini_key"):
                    save_setting(_k, "")
                st.success("已清除保存的钥匙，下次登录需重新输入。")
            st.rerun()

        if has_saved_keys:
            st.caption("🔒 钥匙仅保存在你专属的本地数据库文件中（不会上传到公开仓库，别人看不到）。")

        st.divider()
        st.subheader("买入参数（可调）")
        npr_threshold = st.slider("业绩门槛：净利润同比增长 >（%）", 0, 100, 20, 5,
                                  help="数字越大要求越严。默认20%，即利润要比去年同期涨两成以上。")
        pe_pct_threshold = st.slider("估值门槛：PE 处于近3年分位 <（%）", 0, 100, 50, 5,
                                     help="数字越小要求越便宜。默认50%，即当前估值比近3年一半时间都低。")

        st.divider()
        if st.button("🔄 一键刷新全池数据", use_container_width=True, type="primary"):
            df = load_pool_df()
            if df.empty:
                st.warning("股票池还是空的，请先去『加自选股』添加。")
            else:
                bar = st.progress(0.0, text="开始刷新…")
                total = len(df)
                for i, code in enumerate(df["code"].tolist()):
                    bar.progress((i) / total, text=f"正在刷新 {code} （{i+1}/{total}）…")
                    refresh_one_stock(code)
                bar.progress(1.0, text="刷新完成！")
                st.success(f"已刷新 {total} 只股票的行情/财务/龙虎榜数据。")

        st.divider()
        st.subheader("📋 股票池管理")
        df_side = load_pool_df()
        if df_side.empty:
            st.info("暂无股票。")
        else:
            for _, r in df_side.iterrows():
                c1, c2 = st.columns([3, 1])
                with c1:
                    held = st.checkbox(
                        f"{r['name']}({r['code']}) 我已持仓",
                        value=bool(r["is_holding"]),
                        key=f"hold_{r['code']}",
                    )
                    if held != bool(r["is_holding"]):
                        toggle_holding(r["code"], held)
                        st.rerun()
                with c2:
                    if st.button("删除", key=f"del_{r['code']}"):
                        delete_stock(r["code"])
                        st.rerun()

    # ---------------- 主界面 Tabs ----------------
    tab_decision, tab_add, tab_miner, tab_chip, tab_data = st.tabs(
        ["🎯 今日操作建议", "🛡️ 加自选股", "🕵️‍♂️ 赛道挖掘机", "🖼️ 上传筹码图", "📊 详细数据表"]
    )

    # ===== Tab1：今日操作建议（主页） =====
    with tab_decision:
        st.subheader("🎯 今日操作建议")
        st.caption("系统综合『好公司 + 趋势 + 筹码』给出的明确动作。先填好左侧钥匙、点『一键刷新』、再上传筹码图，建议会更完整。")
        df = load_pool_df()
        if df.empty:
            st.info("股票池还是空的。请到右上角『🛡️ 加自选股』添加你关注的股票。")
        else:
            # 卖出/强烈买入优先排在最前面，方便用户第一眼看到
            order = {"坚决清仓卖出": 0, "强烈买入": 1, "分批建仓买入": 2,
                     "持有移动止盈": 3, "只看不动观望": 4, "待补充筹码数据": 5,
                     "待刷新行情数据": 6}
            cards = []
            for _, r in df.iterrows():
                row = dict(r)
                sig, reason = decide(row, npr_threshold, pe_pct_threshold)
                cards.append((order.get(sig, 9), row, sig, reason))
            # 同一信号档内，紫苏叶评分高（卡位更硬）的票排在前面
            for _, row, sig, reason in sorted(cards, key=lambda x: (x[0], -(x[1].get("serenity_score") or 0))):
                render_signal_card(row, sig, reason)
                # 多空信号对比：一眼看清看涨与看跌两面
                render_bull_bear(row)
                with st.expander("查看这只股的详细数据"):
                    _sc = row.get("serenity_score")
                    st.write({
                        "纳入方式": "👑 人类强制收编（无视AI拒绝）" if row.get("is_override") else "AI 守门员通过",
                        "🌿紫苏叶评分": (f"{int(float(_sc))}/100（{serenity_grade(_sc)}）"
                                      if _sc is not None and not (isinstance(_sc, float) and pd.isna(_sc)) else "未评分（重新研判即可生成）"),
                        "最新收盘价": fmt(row.get("close")),
                        "10日均价线": fmt(row.get("ma10")),
                        "20日均价线": fmt(row.get("ma20")),
                        "30日均价线": fmt(row.get("ma30")),
                        "净利润同比增长(同花顺源)": fmt(row.get("npr_growth"), "%"),
                        "每股收益EPS(同花顺)": fmt(row.get("eps")),
                        "当前PE": fmt(row.get("pe")),
                        "PE近3年分位": fmt(row.get("pe_percentile"), "%"),
                        "市场平均成本": fmt(row.get("avg_cost")),
                        "收盘获利比例": fmt(row.get("profit_ratio"), "%"),
                        "龙虎榜净买入(万元)": fmt(row.get("lhb_net")),
                        "近一交易日是否上龙虎榜": "是" if row.get("lhb_flag") else "否",
                        "今日主力净流入(亿元)": fmt(row.get("main_net_today")),
                        "近5日主力净流入(亿元)": fmt(row.get("main_net_5d")),
                        "融资余额变化(%)": fmt(row.get("margin_chg"), "%"),
                        "股东户数": fmt(row.get("gdhs")),
                        "股东户数较上期变化": (
                            (f"{row.get('gdhs_chg')}%（"
                             + ("↓减少，筹码集中、偏利好"
                                if (row.get('gdhs_chg') is not None and float(row.get('gdhs_chg')) < 0)
                                else ("↑增加，筹码分散、需留意"
                                      if (row.get('gdhs_chg') is not None and float(row.get('gdhs_chg')) > 0)
                                      else "基本持平")) + "）")
                            if row.get("gdhs_chg") is not None else MISSING),
                        "换手率": (f"{row.get('turnover')}%  {turnover_level(row.get('turnover'))[0]}"
                                   if row.get("turnover") is not None else MISSING),
                        "量比（今日/5日均量）": (f"{row.get('vol_ratio')} 倍"
                                                  if row.get("vol_ratio") is not None else MISSING),
                        "筹码-低位单峰密集": _chip_text(row, "chip_single_peak"),
                        "筹码-站上平均成本线": _chip_text(row, "chip_above_avg"),
                        "筹码-高位发散": _chip_text(row, "chip_high_diverge"),
                        "筹码来源": ("自动估算（东财筹码分布，约略）" if (row.get("has_chip") and not row.get("chip_manual"))
                                     else ("看图/人工修正" if row.get("has_chip") else "暂无")),
                    })
                    if row.get("analysis"):
                        st.markdown(f"**AI 选股理由：** {row.get('analysis')}")
                    # 紫苏叶评分分项明细：横向进度条（参考图2风格）
                    render_score_bars(row.get("score_detail"))

                # ===== 🤝 双AI复核：让 DeepSeek + Gemini 一起判断买卖 =====
                with st.expander("🤝 让两个AI（DeepSeek + Gemini）一起复核买卖"):
                    st.caption("上面的结论由『规则引擎』给出。点下面按钮，让两个大模型各自独立再判一次，"
                               "三方一起看更稳。意见一致更可信；分歧时建议谨慎、多看少动。")
                    code_k = row.get("code")
                    if st.button("开始双AI复核", key=f"dual_btn_{code_k}"):
                        with st.spinner("两个AI正在独立研判…"):
                            ds, ds_err, gm, gm_err = dual_decision(
                                row, deepseek_key, deepseek_model, gemini_key, gemini_model,
                                sig, reason, npr_threshold, pe_pct_threshold,
                            )
                        st.session_state[f"dualres_{code_k}"] = (ds, ds_err, gm, gm_err)
                    res = st.session_state.get(f"dualres_{code_k}")
                    if res:
                        ds, ds_err, gm, gm_err = res
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            st.markdown("**🟦 DeepSeek 的意见**")
                            if ds:
                                st.markdown(f"动作：**{ds.get('action', '—')}**"
                                            f"（把握 {round(float(ds.get('confidence') or 0)*100)}%）")
                                st.caption(ds.get("reason", ""))
                            else:
                                st.error(ds_err or "未获取到结果")
                        with cc2:
                            st.markdown("**🟩 Gemini 的意见**")
                            if gm:
                                st.markdown(f"动作：**{gm.get('action', '—')}**"
                                            f"（把握 {round(float(gm.get('confidence') or 0)*100)}%）")
                                st.caption(gm.get("reason", ""))
                            else:
                                st.error(gm_err or "未获取到结果")
                        if ds and gm:
                            a1, a2 = ds.get("action"), gm.get("action")
                            if a1 == a2:
                                st.success(f"✅ 两个AI意见一致：都建议「{a1}」。可结合上方规则引擎结论一起参考。")
                            else:
                                st.warning(f"⚠️ 两个AI意见有分歧：DeepSeek 说「{a1}」，Gemini 说「{a2}」。"
                                           "分歧时更要谨慎，建议多看少动、等信号更明确再决定。")

    # ===== Tab2：加自选股（多轮对话投研 Agent） =====
    with tab_add:
        st.subheader("🛡️ 加自选股 —— 和 AI 投研伙伴聊出来")
        st.caption("直接打一个公司名（如『双环传动』）。AI 不会拿总盘子一刀切地拒绝你，"
                   "而是帮你拆解它的业务线、撇开红海主业、挖出可能符合紫苏叶的『隐藏核心业务』和你商量。"
                   "你说『同意』，它就把这只股按这个逻辑入池。")

        # 顶部工具条：清空对话
        tc1, tc2 = st.columns([1, 1])
        with tc1:
            if st.button("🧹 清空对话，换一只股", use_container_width=True):
                st.session_state["m1_chat"] = []
                st.rerun()

        # 会话历史（既用于界面展示，也作为发给 DeepSeek 的上下文）
        if "m1_chat" not in st.session_state:
            st.session_state["m1_chat"] = []

        # 渲染历史对话
        for msg in st.session_state["m1_chat"]:
            avatar = "🧑‍💼" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        if not st.session_state["m1_chat"]:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("你好！想研究哪只股？直接打公司名或代码（如 **双环传动** 或 **002472**），"
                            "我先帮你把业务线拆开看看，找找有没有藏着的『紫苏叶锚点』。")

        # 聊天输入
        prompt = st.chat_input("输入公司名/代码，或回复我（如『同意』『这个逻辑不硬，看看别的赛道』）…")
        if prompt and prompt.strip():
            st.session_state["m1_chat"].append({"role": "user", "content": prompt.strip()})
            with st.chat_message("user", avatar="🧑‍💼"):
                st.markdown(prompt.strip())

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("AI 正在拆解业务线、寻找紫苏叶锚点…"):
                    ok, data, err = call_deepseek_agent(
                        deepseek_key, deepseek_model, st.session_state["m1_chat"]
                    )
                if not ok:
                    st.error(err)
                    # 失败的这轮用户消息保留，AI 回复不入历史，便于直接重发
                else:
                    reply = data.get("reply") or "（AI 没有给出内容）"
                    st.markdown(reply)
                    st.session_state["m1_chat"].append({"role": "assistant", "content": reply})

                    # 动态入库：仅当 AI 判定用户已同意（ready_to_add）才写库
                    if data.get("ready_to_add"):
                        name = data.get("name") or ""
                        code = data.get("code") or _extract_code(name) or ""
                        anchor = data.get("anchor") or ""
                        analysis = data.get("analysis") or ""
                        if not code:
                            st.warning("我已记下结论，但没能确定股票代码。请直接补一句代码（如 002472），我就入库。")
                        else:
                            layer = data.get("chain_layer") or ""
                            full_analysis = (f"【紫苏叶锚点：{anchor}】" + (f"【{layer}】" if layer else "") + analysis
                                             if (anchor or layer) else analysis)
                            upsert_stock(code, name or code, True, full_analysis)
                            # 算紫苏叶评分并写库（AI 漏给 factors 则不评分，不报错）
                            sc, _ = compute_and_save_score(code, data.get("factors"), data.get("penalties"))
                            score_tip = (f" 紫苏叶评分 {sc}/100（{serenity_grade(sc)}）。" if sc is not None else "")
                            st.success(f"✅ 已按『{anchor or '该逻辑'}』把 {name}({code}) 入池！{score_tip}"
                                       "下一步：左侧『🔄 一键刷新全池数据』拉行情，再到『🖼️ 上传筹码图』补筹码。")

        # ===== 🤝 双AI选股快速把关：DeepSeek + Gemini 各自独立研判是否紫苏叶 =====
        st.divider()
        with st.expander("🤝 双AI选股把关（DeepSeek + Gemini 同时判断是否紫苏叶）"):
            st.caption("上面的对话由 DeepSeek 主导。想让两个AI同时给意见？在这里输入公司名，"
                       "它们各自独立判断是否符合紫苏叶。两个都认 → 更靠谱；意见不一 → 要多想想。")
            ds_name = st.text_input("公司名 / 代码", key="dual_sel_name", placeholder="如 双环传动 或 002472")
            if st.button("让两个AI一起研判", key="dual_sel_btn", type="primary"):
                if not ds_name.strip():
                    st.warning("请先输入公司名或代码。")
                else:
                    with st.spinner("两个AI正在独立研判…"):
                        ds, ds_err, gm, gm_err = dual_select(
                            ds_name.strip(), deepseek_key, deepseek_model, gemini_key, gemini_model
                        )
                    st.session_state["dual_sel_res"] = (ds_name.strip(), ds, ds_err, gm, gm_err)
            sres = st.session_state.get("dual_sel_res")
            if sres:
                sname, ds, ds_err, gm, gm_err = sres
                st.markdown(f"**研判对象：{sname}**")
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.markdown("**🟦 DeepSeek**")
                    if ds:
                        st.markdown("结论：**" + ("✅ 是紫苏叶" if ds.get("is_perilla") else "❌ 不算紫苏叶") + "**")
                        if ds.get("anchor"):
                            st.caption(f"锚点：{ds.get('anchor')}")
                        if ds.get("chain_layer"):
                            st.caption(f"卡位：{ds.get('chain_layer')}")
                        st.caption(ds.get("reason", ""))
                    else:
                        st.error(ds_err or "未获取到结果")
                with sc2:
                    st.markdown("**🟩 Gemini**")
                    if gm:
                        st.markdown("结论：**" + ("✅ 是紫苏叶" if gm.get("is_perilla") else "❌ 不算紫苏叶") + "**")
                        if gm.get("anchor"):
                            st.caption(f"锚点：{gm.get('anchor')}")
                        if gm.get("chain_layer"):
                            st.caption(f"卡位：{gm.get('chain_layer')}")
                        st.caption(gm.get("reason", ""))
                    else:
                        st.error(gm_err or "未获取到结果")
                if ds and gm:
                    if bool(ds.get("is_perilla")) == bool(gm.get("is_perilla")):
                        verdict = "都认为是紫苏叶 ✅" if ds.get("is_perilla") else "都认为不算 ❌"
                        st.success(f"两个AI意见一致：{verdict}。")
                    else:
                        st.warning("两个AI意见不一致，建议回到上方对话框，让 DeepSeek 帮你深入拆解再定。")
                # 一致认可时给个一键入库入口
                if ds and gm and ds.get("is_perilla") and gm.get("is_perilla"):
                    code_guess = _extract_code(sname) or ""
                    anchor = ds.get("anchor") or gm.get("anchor") or ""
                    if code_guess and st.button(f"➕ 两个AI都认可，入池 {sname}", key="dual_sel_add"):
                        full = f"【紫苏叶锚点：{anchor}】DeepSeek与Gemini双模型一致认可。" if anchor else "DeepSeek与Gemini双模型一致认可。"
                        upsert_stock(code_guess, sname, True, full)
                        # 双AI评分取两者平均后算分写库
                        avg_f = _avg_factor_dicts(ds.get("factors"), gm.get("factors"))
                        avg_p = _avg_factor_dicts(ds.get("penalties"), gm.get("penalties"))
                        sc_g, _ = compute_and_save_score(code_guess, avg_f, avg_p)
                        score_tip = (f" 紫苏叶评分 {sc_g}/100（{serenity_grade(sc_g)}）。" if sc_g is not None else "")
                        st.success(f"✅ 已入池 {sname}({code_guess})！{score_tip}请到左侧『🔄 一键刷新全池数据』拉行情。")
                    elif not code_guess:
                        st.info("两个AI都认可。请在上面输入框补上6位代码（如 002472）再研判一次，即可一键入池。")

        # ===== 👑 强制收编（上帝模式）：AI 实在不认时的兜底 =====
        st.divider()
        with st.expander("👑 强制收编（上帝模式）—— AI 死活不认时，手动强行入池"):
            st.caption("如果聊下来 AI 始终不认它是紫苏叶，但你坚持自己的判断，可在此强行纳入。"
                       "强制纳入的股会照常参与行情刷新、筹码分析和买卖建议，并打上『👑 人类强制收编』标签。")
            fc1, fc2 = st.columns([1, 1])
            with fc1:
                f_code = st.text_input("股票代码（6位）", key="force_code", placeholder="如 002472")
            with fc2:
                f_name = st.text_input("股票简称", key="force_name", placeholder="如 双环传动")
            if st.button("👑 强制收编 (Override)", type="primary"):
                code = _extract_code(f_code) or f_code.strip()
                if not code:
                    st.warning("请先填写股票代码。")
                else:
                    force_add_stock(code, f_name.strip() or code,
                                    "用户在对话中坚持纳入，未经AI认可。")
                    st.success(f"👑 已强制收编 {f_name.strip() or code}({code}) 入池！"
                               "请到左侧『🔄 一键刷新全池数据』拉取行情。")

        # ===== 对已入池股票重新研判 =====
        st.divider()
        st.markdown("##### 🔄 对已入池股票重新研判")
        df_re = load_pool_df()
        if df_re.empty:
            st.caption("暂无已入池股票。")
        else:
            sel = st.selectbox("选择股票", options=df_re["code"].tolist(),
                               format_func=lambda c: f"{df_re[df_re['code']==c]['name'].values[0]}({c})")
            if st.button("重新研判这只股"):
                nm = df_re[df_re["code"] == sel]["name"].values[0]
                with st.spinner("重新研判中…"):
                    ok, data, err = call_deepseek_gatekeeper(deepseek_key, deepseek_model, f"{nm} {sel}")
                if ok:
                    layer = data.get("chain_layer") or ""
                    analysis_re = data.get("analysis") or ""
                    full_re = (f"【{layer}】{analysis_re}" if layer else analysis_re)
                    upsert_stock(sel, data.get("name") or nm,
                                 bool(data.get("is_perilla_leaf")), full_re)
                    sc_re, _ = compute_and_save_score(sel, data.get("factors"), data.get("penalties"))
                    score_tip = (f"（🌿 紫苏叶评分 {sc_re}/100 · {serenity_grade(sc_re)}）" if sc_re is not None else "")
                    st.success(f"已更新研判结果。{score_tip}")
                    st.markdown(f"**最新理由：** {analysis_re}")
                else:
                    st.error(err)

    # ===== Tab(新)：AI 热门赛道紫苏叶挖掘机（主动选股） =====
    with tab_miner:
        st.subheader("🕵️‍♂️ AI 热门赛道紫苏叶挖掘机")
        st.caption("不用你想买啥。点一下，AI 自动扫描当下最火的几个硬科技赛道（固态电池、低空经济、商业航天、合成生物…），"
                   "在每个赛道里用『紫苏叶理论』挖出那条别人离不开、卡脖子、玩家极少的底层环节，并推荐对应的 A 股。")

        if st.button("🤖 扫描全市场热门赛道，挖掘紫苏叶", type="primary", use_container_width=True):
            with st.spinner("AI 正在跨赛道深度挖掘（思维链较慢，约需 30–90 秒）…"):
                ok, data, err = call_deepseek_miner(deepseek_key, deepseek_model)
            if not ok:
                st.session_state.pop("miner_result", None)
                st.error(err)
            else:
                _secs = data.get("sectors", [])
                with st.spinner("正在评估各股当前买点（看是否站上均线）…"):
                    eval_buyability_for_sectors(_secs)
                st.session_state["miner_result"] = _secs
                # 新一轮挖掘，清掉旧的『紫苏叶当日精选』（与本轮候选挂钩）
                st.session_state.pop("daily_picks", None)

        sectors = st.session_state.get("miner_result")
        if sectors:
            if st.button("🔄 重新评估各股当前买点（行情会变）", use_container_width=True):
                with st.spinner("正在重新评估各股当前买点…"):
                    eval_buyability_for_sectors(sectors)
                st.session_state["miner_result"] = sectors
                st.rerun()
        if sectors is not None:
            if not sectors:
                st.warning("这次没挖到合适的标的，请再点一次试试。")
            else:
                # 已入池代码，用来判断哪些已经收编过
                _pool_df = load_pool_df()
                pooled = set(_pool_df["code"].astype(str).tolist()) if not _pool_df.empty else set()

                # 证据可信度 → 大白话标签
                def _conf_badge(c):
                    c = (c or "").strip()
                    if "已确认" in c:
                        return "✅ 已确认事实"
                    if "推断" in c:
                        return "🟡 AI推断（仅供参考）"
                    if "待核实" in c or "核实" in c:
                        return "⚠️ 待核实（别全信）"
                    return "🟡 AI推断（仅供参考）"

                # 预先为每只股算紫苏叶评分，并算出每个赛道的平均分用于排序
                for sec in sectors:
                    stks = sec.get("stocks", []) or []
                    scores = []
                    for stk in stks:
                        if stk.get("factors"):
                            sc_v, detail_v = serenity_score(stk.get("factors"), stk.get("penalties"))
                            stk["_score"] = sc_v
                            stk["_detail"] = detail_v
                            scores.append(sc_v)
                        else:
                            stk["_score"] = None
                    # 赛道内排序：先看『今日可买入度』（现在能买的排前），再看紫苏叶评分
                    sec["stocks"] = sorted(
                        stks, key=lambda s: (-(s.get("_buy_score") or -1), -(s.get("_score") or -1)))
                    sec["_avg"] = round(sum(scores) / len(scores)) if scores else None

                # 赛道按平均分高→低排序，让最值得看的赛道排最前
                sectors_sorted = sorted(sectors, key=lambda x: -(x.get("_avg") or -1))

                st.success(f"挖掘完成！AI 扫描出 {len(sectors_sorted)} 个热门赛道，赛道按紫苏叶均分排序；"
                           "每个赛道内已把『现在就能买（站上均线）』的票排在前面。")
                st.caption("⚠️ AI 推荐仅供启发，代码/竞争格局/评分可能有误，收编前请自行核对。"
                           "🌿紫苏叶评分=公司卡位有多硬（基本面）；可买入度=按当天股价判断现在是不是买点（两者分开看）。")

                # ============================================================
                # 当日选股推荐（A 紫苏叶当日精选 + B 热门板块龙头）
                # ============================================================
                st.markdown("---")
                st.markdown("### 🎯 当日选股推荐")
                rally_thr = st.slider(
                    "暴涨过滤：近2-3月涨幅超过多少就排除（避免追高）",
                    min_value=30, max_value=120, value=60, step=5,
                    help="同时作用于『紫苏叶当日精选』和『热门板块龙头』两份名单。"
                         "比如设 60%，意味着近2-3月已涨超 60% 的票会被当作『涨过头』排除。",
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("📅 紫苏叶当日精选（次日买入 Top5）", use_container_width=True):
                        with st.spinner("正在按当天股价+量价+涨幅，从候选里挑次日买入 Top5…"):
                            picks, excluded_cnt = eval_daily_picks(sectors, rally_thr)
                        st.session_state["daily_picks"] = {
                            "picks": picks, "excluded": excluded_cnt, "thr": rally_thr,
                        }
                with col_b:
                    if st.button("🔥 热门板块龙头·技术买点（每板块各3只）", use_container_width=True):
                        with st.spinner("正在抓取热门板块成分股并评估技术买点（联网较多，稍慢）…"):
                            boards, hb_err = eval_hot_board_picks(top_boards=3, per_board=3, rally_threshold=rally_thr)
                        st.session_state["hot_board_picks"] = {"boards": boards, "thr": rally_thr, "err": hb_err}

                # —— A 面板：紫苏叶当日精选 ——
                dp = st.session_state.get("daily_picks")
                if dp is not None:
                    st.markdown("#### 📅 紫苏叶当日精选 · 次日买入（按推荐度排序）")
                    st.caption(f"从挖掘候选里挑出最适合明天买入的票（已排除近2-3月涨幅 > {dp.get('thr')}% 的暴涨股 "
                               f"{dp.get('excluded', 0)} 只）。结合『站上均线/放量确认/紫苏叶卡位/换手不过热』综合评判，仅供参考。")
                    picks = dp.get("picks") or []
                    if not picks:
                        st.info("当前阈值下没有合适的次日买入标的（可能候选都还在均线下方，或都被暴涨过滤了）。可调高滑块再试。")
                    else:
                        for rank, p in enumerate(picks, 1):
                            sc = p.get("score") or 0
                            if sc >= 75:
                                rec_tag, rec_bg = "强烈推荐", "#1a7f37"
                            elif sc >= 55:
                                rec_tag, rec_bg = "可考虑", "#9a6700"
                            else:
                                rec_tag, rec_bg = "谨慎", "#b00020"
                            sr = p.get("serenity")
                            sr_html = (f"<span style='background:#1a3c8c;color:#fff;border-radius:8px;"
                                       f"padding:1px 8px;font-size:13px;margin-left:8px;'>🌿 {sr}/100</span>"
                                       if sr is not None else "")
                            gp = p.get("gain_pct")
                            gp_txt = f"｜近2-3月涨幅 {gp}%" if gp is not None else ""
                            reasons_html = "".join(
                                f"<div style='margin:3px 0;font-size:14px;line-height:1.5;'>→ {r}</div>"
                                for r in (p.get("reasons") or [])
                            ) or "<div style='color:#888;'>技术面中规中矩</div>"
                            st.markdown(
                                f"<div style='background:#f6f9ff;border:1px solid #d6e4ff;border-radius:10px;"
                                f"padding:12px 14px;margin:8px 0;'>"
                                f"<div style='font-size:16px;font-weight:800;color:#1a3c8c;'>"
                                f"#{rank} {p.get('name')}（{p.get('code')}）"
                                f"<span style='background:{rec_bg};color:#fff;border-radius:8px;padding:1px 8px;"
                                f"font-size:13px;margin-left:8px;'>推荐度 {sc}/100 · {rec_tag}</span>{sr_html}</div>"
                                f"<div style='color:#555;font-size:13px;margin-top:4px;'>所属赛道：{p.get('sector')}{gp_txt}</div>"
                                f"<div style='margin-top:6px;'><b>明天买点理由：</b>{reasons_html}</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )

                # —— B 面板：热门板块龙头 ——
                hb = st.session_state.get("hot_board_picks")
                if hb is not None:
                    st.markdown("#### 🔥 热门板块龙头 · 技术买点")
                    st.caption(f"龙头=最近最热门的板块（按当日板块涨幅排序），在每个热门板块成分股里按"
                               f"『技术线 + 当前股价』挑买点，**不要求紫苏叶卡位，仅技术参考**；"
                               f"同样已排除近2-3月涨幅 > {hb.get('thr')}% 的暴涨股。")
                    boards = hb.get("boards") or []
                    if not boards:
                        st.info("暂时没抓到热门板块数据（行情接口可能临时不可用），稍后再点一次试试。")
                        if hb.get("err"):
                            st.caption(f"🔧 诊断信息（截图发我可帮你定位）：{hb.get('err')}")
                    else:
                        for bd in boards:
                            bpct = bd.get("pct")
                            bpct_txt = f"（板块涨 {bpct}%）" if bpct is not None else ""
                            exn = bd.get("excluded", 0)
                            exn_txt = f"｜已排除暴涨股 {exn} 只" if exn else ""
                            with st.expander(f"🔥 {bd.get('board')}{bpct_txt}", expanded=True):
                                bpicks = bd.get("picks") or []
                                if exn_txt:
                                    st.caption(exn_txt.lstrip("｜"))
                                if not bpicks:
                                    st.info("这个板块里暂没挑到合适的技术买点股（可能都偏高或被暴涨过滤了）。")
                                else:
                                    for bp in bpicks:
                                        sc = bp.get("score") or 0
                                        gp = bp.get("gain_pct")
                                        gp_txt = f"｜近2-3月涨幅 {gp}%" if gp is not None else ""
                                        reasons_html = "".join(
                                            f"<div style='margin:3px 0;font-size:14px;line-height:1.5;'>→ {r}</div>"
                                            for r in (bp.get("reasons") or [])
                                        ) or "<div style='color:#888;'>技术面中规中矩</div>"
                                        caution = (bp.get("caution") or "").strip()
                                        caution_html = (f"<div style='color:#b00020;margin-top:4px;font-size:13px;'>"
                                                        f"⚠️ {caution}</div>" if caution else "")
                                        st.markdown(
                                            f"<div style='background:#fff7f0;border:1px solid #ffd8b8;border-radius:10px;"
                                            f"padding:12px 14px;margin:8px 0;'>"
                                            f"<div style='font-size:16px;font-weight:800;color:#9a3412;'>"
                                            f"{bp.get('name')}（{bp.get('code')}）"
                                            f"<span style='background:#9a3412;color:#fff;border-radius:8px;padding:1px 8px;"
                                            f"font-size:13px;margin-left:8px;'>技术买点 {sc}/100</span></div>"
                                            f"<div style='color:#555;font-size:13px;margin-top:4px;'>{gp_txt.lstrip('｜')}</div>"
                                            f"<div style='margin-top:6px;'><b>买入理由：</b>{reasons_html}</div>"
                                            f"{caution_html}"
                                            f"</div>",
                                            unsafe_allow_html=True,
                                        )

                st.markdown("---")
                for si, sec in enumerate(sectors_sorted):
                    sname = sec.get("sector", "未知赛道")
                    avg = sec.get("_avg")
                    avg_tag = f"（赛道均分 {avg}/100 · {serenity_grade(avg)}）" if avg is not None else ""
                    with st.expander(f"🔥 {sname} {avg_tag}", expanded=True):
                        if sec.get("logic"):
                            st.markdown(f"**赛道紫苏叶逻辑：** {sec.get('logic')}")
                        for sti, stk in enumerate(sec.get("stocks", [])):
                            name = stk.get("name") or ""
                            code = str(stk.get("code") or "")
                            sc_v = stk.get("_score")
                            score_html = (f"<span style='background:#1a3c8c;color:#fff;border-radius:8px;"
                                          f"padding:1px 8px;font-size:14px;margin-left:8px;'>🌿 {sc_v}/100 · {serenity_grade(sc_v)}</span>"
                                          if sc_v is not None else
                                          "<span style='background:#bbb;color:#fff;border-radius:8px;padding:1px 8px;font-size:13px;margin-left:8px;'>未评分</span>")
                            # 今日可买入度徽章（按当天股价 vs 均线）
                            bs_v = stk.get("_buy_score")
                            buy_tag = stk.get("_buy_tag") or "买点未知"
                            buy_bg = {"🟢 现在可买": "#1a7f37", "🟡 接近买点": "#9a6700",
                                      "🔴 暂别追": "#b00020"}.get(buy_tag, "#777")
                            buy_html = (f"<span style='background:{buy_bg};color:#fff;border-radius:8px;"
                                        f"padding:1px 8px;font-size:14px;margin-left:8px;'>{buy_tag}"
                                        f"{(' ' + str(bs_v) + '/100') if bs_v is not None else ''}</span>")
                            layer = stk.get("chain_layer") or "—"
                            breaker = stk.get("thesis_breaker") or "—"
                            conf = _conf_badge(stk.get("confidence"))
                            buy_why = stk.get("_buy_why") or ""
                            anti = (stk.get("antipattern_hits") or "").strip()
                            anti_html = ("" if (not anti or anti in ("无", "None", "—"))
                                         else f"<br><span style='color:#9a6700;'><b>⚠️ 伪概念命中：</b>{anti}</span>")
                            st.markdown(
                                f"<div style='background:#f6f9ff;border:1px solid #d6e4ff;"
                                f"border-radius:10px;padding:12px 14px;margin:8px 0;'>"
                                f"<div style='font-size:17px;font-weight:800;color:#1a3c8c;'>"
                                f"📌 {name}（{code}）{score_html}{buy_html}</div>"
                                f"<div style='margin-top:6px;line-height:1.6;'>"
                                f"<b>今日买点：</b>{buy_why}<br>"
                                f"<b>产业链卡位：</b>{layer}<br>"
                                f"<b>卡脖子/底层节点：</b>{stk.get('bottleneck','—')}<br>"
                                f"<b>主要竞争对手：</b>{stk.get('competitors','—')}<br>"
                                f"<b>紫苏叶理由：</b>{stk.get('reason','—')}<br>"
                                f"<b>证据可信度：</b>{conf}{anti_html}<br>"
                                f"<span style='color:#b00020;'><b>⚠️ 这套逻辑的死穴：</b>{breaker}</span>"
                                f"</div></div>",
                                unsafe_allow_html=True,
                            )
                            already = code in pooled
                            btn_key = f"mine_add_{si}_{sti}_{code}"
                            if already:
                                st.caption(f"✅ 『{name}』已在你的股票池里。")
                            elif not code:
                                st.caption("（缺少股票代码，无法一键收编）")
                            else:
                                if st.button(f"➕ 一键收编入库：{name}", key=btn_key):
                                    reason = (f"【AI赛道挖掘·{sname}】{layer}｜卡脖子节点：{stk.get('bottleneck','')}；"
                                              f"竞争对手：{stk.get('competitors','')}；{stk.get('reason','')}"
                                              f"（⚠️逻辑死穴：{breaker}）")
                                    upsert_stock(code, name or code, True, reason)
                                    # 把挖掘时算好的评分一并写库（入池即带分）
                                    compute_and_save_score(code, stk.get("factors"), stk.get("penalties"))
                                    st.success(f"已把 {name}({code}) 收编入池！请到左侧『🔄 一键刷新全池数据』拉行情，"
                                               "再到『🖼️ 上传筹码图』补筹码。")
                                    st.rerun()

    # ===== Tab3：上传筹码图（视觉分析） =====
    with tab_chip:
        st.subheader("🖼️ 上传筹码分布图 —— AI 帮你看图")
        st.caption("从同花顺等软件截一张该股的『筹码分布图』上传，AI 会判断筹码是否健康。这是给出买入建议的必要一步。")
        df_chip = load_pool_df()
        if df_chip.empty:
            st.info("请先到『🛡️ 加自选股』添加股票。")
        else:
            sel = st.selectbox(
                "选择要分析的股票", options=df_chip["code"].tolist(),
                format_func=lambda c: f"{df_chip[df_chip['code']==c]['name'].values[0]}({c})",
                key="chip_sel",
            )
            # 当前股票在库里的数据（手动录入表单用来预填）
            cur = df_chip[df_chip["code"] == sel].iloc[0]

            def _cur_num(col):
                """取当前库里的数值；没有则返回 None（输入框留空）。"""
                v = cur.get(col)
                try:
                    return float(v) if v is not None and pd.notna(v) else None
                except Exception:
                    return None

            up = st.file_uploader(
                "上传筹码分布图（支持手机拍照/相册：png/jpg/heic/webp 等）",
                type=["png", "jpg", "jpeg", "webp", "bmp", "gif", "heic", "heif"],
                help="手机用户：iPhone 默认照片是 HEIC 格式，现在也能直接上传分析。若相册照片选不中，可改用『截图』再上传。",
            )
            if up is not None:
                # 分析按钮紧挨上传框，放在图片预览上方，省得上传后还要往下滚很久
                do_analyze = st.button("🤖 让 AI 分析这张图")
                if do_analyze:
                    with st.spinner("视觉模型分析中…"):
                        ok, data, err = call_gemini_chip(
                            gemini_key, gemini_model, up.getvalue(), getattr(up, "type", None)
                        )
                    if not ok:
                        st.error(err)
                    else:
                        # 筹码结论
                        save = {
                            "chip_single_peak": 1 if data.get("is_single_peak_low") else 0,
                            "chip_above_avg": 1 if data.get("above_avg_cost") else 0,
                            "chip_high_diverge": 1 if data.get("is_high_diverge") else 0,
                            "chip_confidence": float(data.get("confidence") or 0),
                            # 看图结果更精准，标记为优先（chip_manual=1）：之后『一键刷新』里的自动筹码不会覆盖它
                            "chip_manual": 1,
                            "has_chip": 1,
                        }
                        # 从图中读到的行情/估值/资金（作为 akshare 失败时的备用来源）：只在读到数字时才写入
                        img_data = {
                            "close": _to_float_pct(data.get("close_price")),
                            "ma10": _to_float_pct(data.get("ma10")),
                            "ma20": _to_float_pct(data.get("ma20")),
                            "ma30": _to_float_pct(data.get("ma30")),
                            "pe": _to_float_pct(data.get("pe")),
                            "avg_cost": _to_float_pct(data.get("avg_cost")),
                            "profit_ratio": _to_float_pct(data.get("profit_ratio")),
                            "lhb_net": _to_float_pct(data.get("lhb_net_wan")),
                            "main_net_today": _to_float_pct(data.get("main_net_today_yi")),
                            "main_net_5d": _to_float_pct(data.get("main_net_5d_yi")),
                            "margin_chg": _to_float_pct(data.get("margin_change_pct")),
                        }
                        for k, v in img_data.items():
                            if v is not None:
                                save[k] = v
                        # 图里读到龙虎榜净额，说明它上了龙虎榜
                        if img_data["lhb_net"] is not None:
                            save["lhb_flag"] = 1
                        update_fields(sel, save)
                        st.success("分析完成，结果已保存！")
                        st.markdown(f"**大白话解读：** {data.get('explain', '')}")
                        st.write({
                            "低位单峰密集": "是" if data.get("is_single_peak_low") else "否",
                            "站上平均成本线": "是" if data.get("above_avg_cost") else "否",
                            "高位发散": "是" if data.get("is_high_diverge") else "否",
                            "图中现价": img_data["close"] if img_data["close"] is not None else "未读到",
                            "图中MA20 / MA30": f"{img_data['ma20']} / {img_data['ma30']}",
                            "图中PE": img_data["pe"] if img_data["pe"] is not None else "未读到",
                            "图中平均成本": img_data["avg_cost"] if img_data["avg_cost"] is not None else "未读到",
                            "图中获利比例%": img_data["profit_ratio"] if img_data["profit_ratio"] is not None else "未读到",
                            "图中龙虎榜净买入(万)": img_data["lhb_net"] if img_data["lhb_net"] is not None else "未读到",
                            "图中今日主力净流入(亿)": img_data["main_net_today"] if img_data["main_net_today"] is not None else "未读到",
                            "图中近5日主力净流入(亿)": img_data["main_net_5d"] if img_data["main_net_5d"] is not None else "未读到",
                            "图中融资余额变化%": img_data["margin_chg"] if img_data["margin_chg"] is not None else "未读到",
                            "置信度": f"{round(float(data.get('confidence') or 0)*100)}%",
                        })
                        if img_data["close"] is None or img_data["ma30"] is None:
                            st.info("提示：这张图里没读全『现价/均线』数字。如果你的图上有这些数字，请换一张更清晰、能看到现价和均线数值的截图，系统就能直接给出买卖建议。")

                # ===== ✍️ 手动录入关键数据（紧挨分析按钮，AI 读图失败 / akshare 抓不到时用）=====
                with st.expander("✍️ 手动录入关键数据（AI 没读出或 akshare 没抓到时，自己填）", expanded=False):
                    st.caption("如市盈率 PE、均线等数字 AI 没读出来、akshare 也没抓到，可在这里自己填。"
                               "**留空的格子不会改动原有数据**；填了的会直接覆盖保存。")
                    with st.form("manual_data_form"):
                        fa, fb, fc = st.columns(3)
                        with fa:
                            in_close = st.number_input("收盘价", value=_cur_num("close"), step=0.01, format="%.2f")
                            in_ma10 = st.number_input("MA10（10日均价）", value=_cur_num("ma10"), step=0.01, format="%.2f")
                            in_ma20 = st.number_input("MA20（20日均价）", value=_cur_num("ma20"), step=0.01, format="%.2f")
                            in_ma30 = st.number_input("MA30（30日均价）", value=_cur_num("ma30"), step=0.01, format="%.2f")
                        with fb:
                            in_pe = st.number_input("市盈率 PE", value=_cur_num("pe"), step=0.01, format="%.2f")
                            in_pepct = st.number_input("PE近3年分位（%）", value=_cur_num("pe_percentile"), step=0.1, format="%.1f")
                            in_npr = st.number_input("净利润同比增长（%）", value=_cur_num("npr_growth"), step=0.1, format="%.1f")
                            in_eps = st.number_input("每股收益 EPS", value=_cur_num("eps"), step=0.01, format="%.2f")
                            in_avg = st.number_input("平均成本", value=_cur_num("avg_cost"), step=0.01, format="%.2f")
                        with fc:
                            in_profit = st.number_input("获利比例（%）", value=_cur_num("profit_ratio"), step=0.1, format="%.1f")
                            in_lhb = st.number_input("龙虎榜净买入（万元，净卖出填负）", value=_cur_num("lhb_net"), step=1.0, format="%.0f")
                            in_m1 = st.number_input("今日主力净流入（亿元，净流出填负）", value=_cur_num("main_net_today"), step=0.01, format="%.2f")
                            in_m5 = st.number_input("近5日主力净流入（亿元，净流出填负）", value=_cur_num("main_net_5d"), step=0.01, format="%.2f")
                        in_margin = st.number_input("融资余额变化（%，减少填负）", value=_cur_num("margin_chg"), step=0.01, format="%.2f")

                        submitted = st.form_submit_button("💾 保存手动录入的数据", type="primary")
                        if submitted:
                            mapping = {
                                "close": in_close, "ma10": in_ma10, "ma20": in_ma20, "ma30": in_ma30,
                                "pe": in_pe, "pe_percentile": in_pepct, "npr_growth": in_npr, "eps": in_eps,
                                "avg_cost": in_avg, "profit_ratio": in_profit, "lhb_net": in_lhb,
                                "main_net_today": in_m1, "main_net_5d": in_m5, "margin_chg": in_margin,
                            }
                            # 只保存"填了"的格子（留空=None=不改动）
                            to_save = {k: v for k, v in mapping.items() if v is not None}
                            if not to_save:
                                st.warning("你没有填写任何数字。")
                            else:
                                # 填了龙虎榜净额，顺手把"上龙虎榜"标记打上
                                if "lhb_net" in to_save:
                                    to_save["lhb_flag"] = 1
                                update_fields(sel, to_save)
                                st.success(f"已手动保存 {len(to_save)} 项数据，将直接参与买卖决策。"
                                           "可到『🎯 今日操作建议』查看更新后的结论。")

                # 图片预览放在按钮/结果下方，作为参考（HEIC 等格式浏览器可能无法预览，不影响分析）
                try:
                    st.image(up, caption="你上传的筹码图", use_container_width=True)
                except Exception:
                    st.caption("（这张图是 HEIC 等手机格式，浏览器无法直接预览，但不影响上方 AI 分析。）")

            # 人工修正区（人工值优先于模型值）
            st.divider()
            st.markdown("##### ✍️ 人工修正（若你觉得 AI 看错了，可在此手动调整）")
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                m_single = st.checkbox("低位单峰密集", value=bool(cur["chip_single_peak"]), key="m_single")
            with mc2:
                m_above = st.checkbox("站上平均成本线", value=bool(cur["chip_above_avg"]), key="m_above")
            with mc3:
                m_div = st.checkbox("高位发散", value=bool(cur["chip_high_diverge"]), key="m_div")
            if st.button("保存人工修正"):
                update_fields(sel, {
                    "chip_single_peak": 1 if m_single else 0,
                    "chip_above_avg": 1 if m_above else 0,
                    "chip_high_diverge": 1 if m_div else 0,
                    "chip_manual": 1,
                    "has_chip": 1,
                })
                st.success("人工修正已保存（将优先于 AI 判断使用）。")

    # ===== Tab4：详细数据表（进阶） =====
    with tab_data:
        st.subheader("📊 详细数据表（给想看细节的你）")
        df = load_pool_df()
        if df.empty:
            st.info("股票池为空。")
        else:
            show = df.copy()
            # 生成决策列
            sigs, reasons = [], []
            for _, r in show.iterrows():
                s, why = decide(dict(r), npr_threshold, pe_pct_threshold)
                sigs.append(f"{SIGNAL_STYLE.get(s, {}).get('emoji','')} {s}")
                reasons.append(why)
            show["操作建议"] = sigs
            show["建议原因"] = reasons

            # 把 0/1 这类布尔列转成"是/否"，更直观
            def _yn(v):
                return "是" if (v in (1, "1", True) or v == 1) else "否"
            for bcol in ["is_holding", "is_override", "lhb_flag",
                         "chip_single_peak", "chip_above_avg", "chip_high_diverge", "has_chip"]:
                if bcol in show.columns:
                    show[bcol] = show[bcol].apply(_yn)

            rename = {
                "code": "代码", "name": "名称", "is_holding": "已持仓",
                "is_override": "👑强制收编", "serenity_score": "🌿紫苏叶评分",
                "npr_growth": "净利润同比增长%", "eps": "每股收益EPS", "pe": "PE", "pe_percentile": "PE近3年分位%",
                "close": "收盘价", "ma10": "MA10(10日均价)", "ma20": "MA20(20日均价)",
                "ma30": "MA30(30日均价)", "lhb_flag": "上龙虎榜", "lhb_net": "龙虎榜净买入(万)",
                "avg_cost": "平均成本", "profit_ratio": "获利比例%",
                "main_net_today": "今日主力净流入(亿)", "main_net_5d": "近5日主力净流入(亿)",
                "margin_chg": "融资余额变化%",
                "gdhs": "股东户数", "gdhs_chg": "股东户数变化%",
                "turnover": "换手率%", "vol_ratio": "量比",
                "chip_single_peak": "低位单峰密集", "chip_above_avg": "站上成本线",
                "chip_high_diverge": "高位发散", "chip_confidence": "视觉置信度",
                "has_chip": "已有筹码分析", "updated_at": "更新时间",
            }
            cols = ["code", "name", "操作建议", "建议原因", "serenity_score", "is_override", "is_holding", "close",
                    "ma10", "ma20", "ma30", "npr_growth", "eps", "pe", "pe_percentile",
                    "avg_cost", "profit_ratio", "lhb_flag", "lhb_net",
                    "main_net_today", "main_net_5d", "margin_chg", "gdhs", "gdhs_chg",
                    "turnover", "vol_ratio",
                    "chip_single_peak", "chip_above_avg", "chip_high_diverge",
                    "chip_confidence", "has_chip", "updated_at"]
            cols = [c for c in cols if c in show.columns or c in ("操作建议", "建议原因")]
            show = show[cols].rename(columns=rename)

            # 列宽配置：把会被截断的长文字列设宽，并允许悬停看全
            col_config = {
                "建议原因": st.column_config.TextColumn("建议原因", width="large"),
                "操作建议": st.column_config.TextColumn("操作建议", width="medium"),
                "名称": st.column_config.TextColumn("名称", width="small"),
            }
            # 高度自适应：把所有股票一次展示完，不要内部滚动条藏行
            table_h = min(680, 80 + 38 * max(1, len(show)))
            st.dataframe(
                show,
                use_container_width=True,
                hide_index=True,
                height=table_h,
                column_config=col_config,
            )
            st.caption("💡 表格里被截断的文字，把鼠标放上去会显示全文；也可以左右拖动表格、"
                       "或点右上角放大按钮全屏看。想看完整段落，请展开下方『完整文字视图』。")

            # 下载完整数据（含所有文字）为 CSV
            st.download_button(
                "⬇️ 导出完整数据表（CSV）",
                data=show.to_csv(index=False).encode("utf-8-sig"),
                file_name="股票池详细数据.csv",
                mime="text/csv",
            )

            # 完整文字视图：逐只股纵向展示，文字/数据一个都不截断
            with st.expander("📖 完整文字视图（每只股的全部文字与数据，绝不截断）", expanded=False):
                for _, r in df.iterrows():
                    sig, why = decide(dict(r), npr_threshold, pe_pct_threshold)
                    emoji = SIGNAL_STYLE.get(sig, {}).get("emoji", "")
                    ov = " 👑人类强制收编" if r.get("is_override") in (1, "1", True) else ""
                    st.markdown(f"#### {r.get('name','')}（{r.get('code','')}）{ov}")
                    st.markdown(f"**操作建议：** {emoji} {sig}")
                    st.markdown(f"**建议原因：** {why}")
                    if r.get("analysis"):
                        st.markdown(f"**AI 选股理由 / 紫苏叶锚点：** {r.get('analysis')}")
                    st.markdown(
                        f"- 收盘价 {fmt(r.get('close'))}｜MA10 {fmt(r.get('ma10'))}｜"
                        f"MA20 {fmt(r.get('ma20'))}｜MA30 {fmt(r.get('ma30'))}\n"
                        f"- 净利润同比增长 {fmt(r.get('npr_growth'),'%')}｜PE {fmt(r.get('pe'))}｜"
                        f"PE近3年分位 {fmt(r.get('pe_percentile'),'%')}\n"
                        f"- 平均成本 {fmt(r.get('avg_cost'))}｜获利比例 {fmt(r.get('profit_ratio'),'%')}｜"
                        f"龙虎榜净买入 {fmt(r.get('lhb_net'))}万\n"
                        f"- 今日主力净流入 {fmt(r.get('main_net_today'))}亿｜"
                        f"近5日主力净流入 {fmt(r.get('main_net_5d'))}亿｜"
                        f"融资余额变化 {fmt(r.get('margin_chg'),'%')}\n"
                        f"- 更新时间：{r.get('updated_at') or '—'}"
                    )
                    st.divider()

            st.caption("名词解释：MA = 均价线（最近N天平均成本）；PE = 市盈率（越低越便宜）；"
                       "PE近3年分位 = 当前估值在近3年里的高低位置（越低越便宜）；"
                       "戴维斯双击 = 业绩大涨 + 估值偏低 的双重利好。")


def _chip_text(row, key):
    """把筹码布尔字段转成中文展示；无数据显示『未分析』。"""
    if not row.get("has_chip"):
        return "未分析（请上传筹码图）"
    return "是" if row.get(key) else "否"


def _extract_code(text):
    """从文本中提取6位股票代码；提取不到返回 None。"""
    import re
    if not text:
        return None
    m = re.search(r"\d{6}", str(text))
    return m.group(0) if m else None


if __name__ == "__main__":
    main()
