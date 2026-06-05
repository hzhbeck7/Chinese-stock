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

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perilla_stock.db")  # SQLite 数据库文件
MISSING = "数据缺失"          # 取数失败时的统一占位
HTTP_TIMEOUT = 30             # 网络请求超时（秒）

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
    """初始化数据库，创建股票池表（若不存在）。"""
    conn = sqlite3.connect(DB_PATH)
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
            updated_at       TEXT                -- 最后更新时间
        )
        """
    )
    conn.commit()
    conn.close()


def db_conn():
    """返回数据库连接（行可按列名访问）。"""
    conn = sqlite3.connect(DB_PATH)
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

PERILLA_SYSTEM_PROMPT = """你是一位顶级的A股硬科技产业链研究专家，精通"紫苏叶理论"。
"紫苏叶公司"的严格标准（必须同时满足）：
1. 处于产业链的深层节点（Layer3 及以下，即底层硬件、核心材料、关键设备等"卖水人"角色），而非终端品牌或应用层。
2. 产品/技术不可替代，具有很高的技术壁垒或专利护城河。
3. 处于寡头垄断格局（全球或国内有效竞争对手 <= 3 家）。

请基于你的知识，判断用户给出的公司是否符合"紫苏叶公司"标准。
你必须只返回一个 JSON 对象，不要任何额外文字、不要markdown代码块标记，格式严格如下：
{"is_perilla_leaf": true 或 false, "name": "公司中文简称", "analysis": "用通俗易懂的大白话解释判断理由，150字以内，让股票小白也能看懂"}
"""


def call_deepseek_gatekeeper(api_key, model, user_input):
    """
    调用 DeepSeek 进行紫苏叶研判。
    返回 (ok, result_dict, err_msg)。
    result_dict 含 is_perilla_leaf / name / analysis。
    """
    if not api_key:
        return False, None, "未填写 DeepSeek API Key（请在左侧边栏填写）。"

    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=HTTP_TIMEOUT)
        if resp.status_code != 200:
            return False, None, f"DeepSeek 接口返回错误 {resp.status_code}：{resp.text[:200]}"
        content = resp.json()["choices"][0]["message"]["content"]
        data = _safe_parse_json(content)
        if data is None or "is_perilla_leaf" not in data:
            return False, None, f"AI 返回内容无法解析为标准结果：{content[:200]}"
        return True, data, ""
    except requests.exceptions.Timeout:
        return False, None, "调用 DeepSeek 超时，请检查网络后重试。"
    except requests.exceptions.RequestException as e:
        return False, None, f"网络请求异常：{e}"
    except Exception as e:
        return False, None, f"未知错误：{e}"


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


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_price_ma(code):
    """
    拉取日线行情，计算最新收盘价与 MA10/MA20/MA30。
    成功返回 dict；失败则抛出异常（注意：Streamlit 不会缓存抛异常的结果，
    因此下次点『刷新』会自动重试，而不会被旧的失败结果卡住）。
    """
    ak = _get_akshare()
    if ak is None:
        raise RuntimeError("akshare 未安装")

    end = datetime.date.today().strftime("%Y%m%d")
    start = (datetime.date.today() - datetime.timedelta(days=200)).strftime("%Y%m%d")
    last_err = "未知原因"
    # 网络可能抽风，最多重试 3 次
    for _ in range(3):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                    start_date=start, end_date=end, adjust="qfq")
            if df is None or df.empty:
                last_err = "接口返回空数据"
                continue
            df = df.sort_values("日期")
            close = df["收盘"].astype(float)
            return {
                "close": round(float(close.iloc[-1]), 2),
                "ma10": round(float(close.rolling(10).mean().iloc[-1]), 2) if len(close) >= 10 else None,
                "ma20": round(float(close.rolling(20).mean().iloc[-1]), 2) if len(close) >= 20 else None,
                "ma30": round(float(close.rolling(30).mean().iloc[-1]), 2) if len(close) >= 30 else None,
            }
        except Exception as e:
            last_err = str(e)
            continue
    raise RuntimeError(f"行情数据获取失败：{last_err}")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_npr_growth(code):
    """
    拉取净利润同比增长率（%）。主用 stock_financial_abstract，失败返回 None。
    （注：作为需求中 EPS 增长率的代理指标，口径为净利润同比。）
    """
    ak = _get_akshare()
    if ak is None:
        return None
    # 方案A：财务摘要
    try:
        df = ak.stock_financial_abstract(symbol=code)
        if df is not None and not df.empty:
            # 不同版本列名可能不同，模糊查找含"净利润"且含"同比"的行/列
            txt_cols = [c for c in df.columns if isinstance(c, str)]
            # 形态1：含"指标"列的长表
            if "指标" in df.columns:
                mask = df["指标"].astype(str).str.contains("净利润") & \
                       df["指标"].astype(str).str.contains("同比|增长")
                sub = df[mask]
                if not sub.empty:
                    # 取最右侧（最新）一个非空数值
                    row = sub.iloc[0]
                    for v in reversed(list(row.values)):
                        val = _to_float_pct(v)
                        if val is not None:
                            return val
    except Exception:
        pass
    # 方案B：业绩报表
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
            row = df[df[code_col].astype(str).str.zfill(6) == code]
            if not row.empty:
                for c in df.columns:
                    if "净利润" in str(c) and ("同比" in str(c) or "增长" in str(c)):
                        val = _to_float_pct(row.iloc[0][c])
                        if val is not None:
                            return val
    except Exception:
        pass
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pe_percentile(code):
    """
    拉取历史 PE 序列，计算当前 PE 及其近3年分位（%）。
    返回 (pe, pe_percentile) ；失败返回 (None, None)。
    """
    ak = _get_akshare()
    if ak is None:
        return None, None
    df = None
    for _ in range(2):  # legulegu 源不稳，重试 2 次
        try:
            df = ak.stock_a_indicator_lg(symbol=code)  # legulegu 历史估值
            if df is not None and not df.empty:
                break
        except Exception:
            df = None
            continue
    try:
        if df is None or df.empty:
            return None, None
        # 找到 PE 列与日期列
        pe_col = None
        for c in df.columns:
            if str(c).lower() in ("pe", "pe_ttm") or "市盈率" in str(c):
                pe_col = c
                break
        if pe_col is None:
            return None, None
        date_col = "trade_date" if "trade_date" in df.columns else df.columns[0]
        df = df[[date_col, pe_col]].dropna()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna().sort_values(date_col)
        # 近3年
        cutoff = pd.Timestamp(datetime.date.today() - datetime.timedelta(days=365 * 3))
        recent = df[df[date_col] >= cutoff]
        if recent.empty:
            recent = df
        pe_series = recent[pe_col].astype(float)
        cur_pe = float(pe_series.iloc[-1])
        # 分位：当前 PE 在历史序列中的百分位排名
        pct = round(float((pe_series <= cur_pe).mean() * 100), 1)
        return round(cur_pe, 2), pct
    except Exception:
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

    # 行情与均线（失败不写入，留空显示"数据缺失"；因抛异常未被缓存，下次刷新会重试）
    try:
        ma = fetch_price_ma(code)
        for k in ("close", "ma10", "ma20", "ma30"):
            fields[k] = ma.get(k)
    except Exception:
        for k in ("close", "ma10", "ma20", "ma30"):
            fields[k] = None

    # 净利润同比增长
    fields["npr_growth"] = fetch_npr_growth(code)

    # PE 与分位
    pe, pe_pct = fetch_pe_percentile(code)
    fields["pe"] = pe
    fields["pe_percentile"] = pe_pct

    # 龙虎榜
    lhb = fetch_lhb_flag(code)
    fields["lhb_flag"] = 1 if lhb else 0

    update_fields(code, fields)


# ============================================================================
# 模块3：AI 视觉分析师（Gemini 筹码分布图分析）
# ============================================================================

CHIP_VISION_PROMPT = """你是一位精通筹码分布（成本分布）分析的A股技术专家。
我会给你一张同花顺风格的"筹码分布图"。请分析后只返回一个 JSON 对象，不要任何额外文字、不要markdown标记：
{
 "is_single_peak_low": true 或 false,   // 是否为"低位单峰密集"（筹码高度集中在当前价附近的低位区域）
 "above_avg_cost": true 或 false,       // 当前股价是否站上了"平均成本线"
 "is_high_diverge": true 或 false,      // 是否为"高位发散"（筹码在高位分散、获利盘巨大、有派发风险）
 "confidence": 0.0 到 1.0 之间的小数,    // 你对本次判断的置信度
 "explain": "用大白话解释当前筹码状态，100字以内，让股票小白能看懂"
}
"""


def call_gemini_chip(api_key, model, image_bytes):
    """
    调用 Gemini 多模态分析筹码图。
    返回 (ok, result_dict, err_msg)。
    """
    if not _GENAI_OK:
        return False, None, "未安装 google-generativeai 库，请先 pip install。"
    if not _PIL_OK:
        return False, None, "未安装 Pillow 库，请先 pip install Pillow。"
    if not api_key:
        return False, None, "未填写 Gemini API Key（请在左侧边栏填写）。"
    try:
        genai.configure(api_key=api_key)
        img = Image.open(BytesIO(image_bytes))
        gmodel = genai.GenerativeModel(model or "gemini-1.5-flash")
        resp = gmodel.generate_content(
            [CHIP_VISION_PROMPT, img],
            request_options={"timeout": HTTP_TIMEOUT},
        )
        text = getattr(resp, "text", None) or ""
        data = _safe_parse_json(text)
        if data is None or "is_single_peak_low" not in data:
            return False, None, f"视觉模型返回无法解析：{text[:200]}"
        return True, data, ""
    except Exception as e:
        return False, None, f"调用 Gemini 失败：{e}"


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

    # 1) 持仓者破位风控（最高优先）
    if is_holding and (close < ma30 or high_diverge):
        if close < ma30:
            return "坚决清仓卖出", f"你持有的这只股，股价（{close}）已跌破30日均价线（{ma30}）这条重要生命线，按纪律应坚决离场止损。"
        return "坚决清仓卖出", "你持有的这只股出现筹码『高位发散』，主力可能在高位派发，建议坚决离场。"

    # 2) 视觉必需 gate：没有筹码分析就不出完整买入建议
    if not has_chip:
        return "待补充筹码数据", "还没上传这只股的『筹码分布图』。请到『上传筹码图』页面上传后，系统才能给出完整买卖建议。"

    # 3) 紫苏叶逻辑成立
    if is_perilla:
        if ma20 is not None and close > ma20 and close > ma30:
            # 站上均线，处于多头
            if is_holding:
                return "持有移动止盈", f"股价（{close}）稳稳站在均价线之上，趋势健康，继续持有。移动止盈提示：跌破10日均价线（{row.get('ma10')}）可考虑减仓，跌破30日均价线（{ma30}）则清仓。"
            else:
                davis = is_davis_double(row, npr_threshold, pe_pct_threshold)
                if single_peak or davis:
                    why = []
                    if single_peak:
                        why.append("筹码处于低位单峰密集（成本集中、抛压小）")
                    if davis:
                        why.append("业绩大涨且估值偏低（戴维斯双击）")
                    return "强烈买入", "符合紫苏叶好公司，且股价站上均线，又叠加" + "、".join(why) + "，是难得的好买点，可分批建仓。"
                return "分批建仓买入", f"这是符合紫苏叶标准的好公司，股价（{close}）已站上20日和30日均价线，进入右侧上涨，可分批建仓买入。"
        elif close < ma30:
            return "只看不动观望", f"好公司，但股价（{close}）还在30日均价线（{ma30}）下方，处于左侧寻底阶段，先观望，等它站稳均线再说。"
        else:
            return "只看不动观望", f"好公司，股价（{close}）在30日均价线之上但还没站上20日线（{ma20}），趋势未完全走强，先观望。"

    # 兜底
    return "只看不动观望", "暂不满足明确的买入或卖出条件，保持观望。"


# ============================================================================
# 模块5：Streamlit 小白友好 UI
# ============================================================================

def render_signal_card(row, signal_key, reason):
    """渲染一张大色块操作建议卡片。"""
    style = SIGNAL_STYLE.get(signal_key, SIGNAL_STYLE["只看不动观望"])
    name = row.get("name") or ""
    code = row.get("code") or ""
    hold_tag = "（已持仓）" if row.get("is_holding") else "（未持仓）"
    st.markdown(
        f"""
        <div style="background:{style['color']};color:{style['text']};
                    padding:18px 20px;border-radius:14px;margin-bottom:14px;">
          <div style="font-size:22px;font-weight:800;">
            {style['emoji']} {name} {code} {hold_tag} —— {signal_key}
          </div>
          <div style="font-size:16px;margin-top:8px;line-height:1.6;">{reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt(v, suffix=""):
    """格式化展示：None -> 数据缺失。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return MISSING
    return f"{v}{suffix}"


def main():
    st.set_page_config(page_title="紫苏叶 AI 投研系统", page_icon="🌿", layout="wide")
    init_db()

    st.title("🌿 硬科技【紫苏叶 AI 投研与综合择时系统】")
    st.caption("找到产业链最底层、别人离不开的好公司，在合适的时机告诉你该买、该卖还是该等。")

    # ---------------- 侧边栏 ----------------
    with st.sidebar:
        st.header("⚙️ 设置")
        st.subheader("AI 钥匙（API Key）")
        deepseek_key = st.text_input("DeepSeek API Key", type="password",
                                     help="用于『AI 选股研判』。从 deepseek.com 申请。只存在本次会话，不会保存到文件。")
        deepseek_model = st.text_input("DeepSeek 模型名", value="deepseek-chat",
                                       help="一般保持默认即可。")
        gemini_key = st.text_input("Gemini API Key", type="password",
                                   help="用于『筹码分布图』看图分析。从 Google AI Studio 申请。只存在本次会话。")
        gemini_model = st.text_input("Gemini 模型名", value="gemini-1.5-flash",
                                     help="一般保持默认即可。")

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
    tab_decision, tab_add, tab_chip, tab_data = st.tabs(
        ["🎯 今日操作建议", "🛡️ 加自选股", "🖼️ 上传筹码图", "📊 详细数据表"]
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
            for _, row, sig, reason in sorted(cards, key=lambda x: x[0]):
                render_signal_card(row, sig, reason)
                with st.expander("查看这只股的详细数据"):
                    st.write({
                        "最新收盘价": fmt(row.get("close")),
                        "10日均价线": fmt(row.get("ma10")),
                        "20日均价线": fmt(row.get("ma20")),
                        "30日均价线": fmt(row.get("ma30")),
                        "净利润同比增长(口径:净利润，代理EPS)": fmt(row.get("npr_growth"), "%"),
                        "当前PE": fmt(row.get("pe")),
                        "PE近3年分位": fmt(row.get("pe_percentile"), "%"),
                        "近一交易日是否上龙虎榜": "是" if row.get("lhb_flag") else "否",
                        "筹码-低位单峰密集": _chip_text(row, "chip_single_peak"),
                        "筹码-站上平均成本线": _chip_text(row, "chip_above_avg"),
                        "筹码-高位发散": _chip_text(row, "chip_high_diverge"),
                    })
                    if row.get("analysis"):
                        st.markdown(f"**AI 选股理由：** {row.get('analysis')}")

    # ===== Tab2：加自选股（守门员） =====
    with tab_add:
        st.subheader("🛡️ 加自选股 —— AI 帮你把关")
        st.caption("输入股票代码或名称，AI 会判断它是不是『产业链最底层、别人离不开』的好公司。是，才会加入你的池子。")
        col1, col2 = st.columns([3, 1])
        with col1:
            user_input = st.text_input("股票代码 / 简称", placeholder="例如：北方华创 或 002371")
        with col2:
            st.write("")
            st.write("")
            do_judge = st.button("🔍 让 AI 研判", use_container_width=True)

        # 已入池股票的"重新研判"区
        if do_judge and user_input.strip():
            with st.spinner("AI 正在深度研判中…"):
                ok, data, err = call_deepseek_gatekeeper(deepseek_key, deepseek_model, user_input.strip())
            if not ok:
                st.error(err)
            else:
                is_perilla = bool(data.get("is_perilla_leaf"))
                name = data.get("name") or user_input.strip()
                analysis = data.get("analysis") or ""
                code = _extract_code(user_input) or _extract_code(name) or user_input.strip()
                if is_perilla:
                    upsert_stock(code, name, True, analysis)
                    st.success(f"✅ 通过！『{name}』符合紫苏叶标准，已加入股票池。")
                    st.markdown(f"**AI 理由：** {analysis}")
                    st.info("下一步：点左侧『🔄 一键刷新全池数据』拉取行情，再到『🖼️ 上传筹码图』补充筹码分析。")
                else:
                    st.warning(f"❌ 未通过：『{name}』不符合紫苏叶标准，不加入池子。")
                    st.markdown(f"**AI 理由：** {analysis}")
        elif do_judge:
            st.warning("请先输入股票代码或简称。")

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
                    upsert_stock(sel, data.get("name") or nm,
                                 bool(data.get("is_perilla_leaf")), data.get("analysis") or "")
                    st.success("已更新研判结果。")
                    st.markdown(f"**最新理由：** {data.get('analysis')}")
                else:
                    st.error(err)

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
            up = st.file_uploader("上传筹码分布图（png/jpg）", type=["png", "jpg", "jpeg"])
            if up is not None:
                st.image(up, caption="你上传的筹码图", use_container_width=True)
                if st.button("🤖 让 AI 分析这张图"):
                    with st.spinner("视觉模型分析中…"):
                        ok, data, err = call_gemini_chip(gemini_key, gemini_model, up.getvalue())
                    if not ok:
                        st.error(err)
                    else:
                        update_fields(sel, {
                            "chip_single_peak": 1 if data.get("is_single_peak_low") else 0,
                            "chip_above_avg": 1 if data.get("above_avg_cost") else 0,
                            "chip_high_diverge": 1 if data.get("is_high_diverge") else 0,
                            "chip_confidence": float(data.get("confidence") or 0),
                            "chip_manual": 0,
                            "has_chip": 1,
                        })
                        st.success("分析完成，结果已保存！")
                        st.markdown(f"**大白话解读：** {data.get('explain', '')}")
                        st.write({
                            "低位单峰密集": "是" if data.get("is_single_peak_low") else "否",
                            "站上平均成本线": "是" if data.get("above_avg_cost") else "否",
                            "高位发散": "是" if data.get("is_high_diverge") else "否",
                            "置信度": f"{round(float(data.get('confidence') or 0)*100)}%",
                        })

            # 人工修正区（人工值优先于模型值）
            st.divider()
            st.markdown("##### ✍️ 人工修正（若你觉得 AI 看错了，可在此手动调整）")
            cur = df_chip[df_chip["code"] == sel].iloc[0]
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
            rename = {
                "code": "代码", "name": "名称", "is_holding": "已持仓",
                "npr_growth": "净利润同比增长%", "pe": "PE", "pe_percentile": "PE近3年分位%",
                "close": "收盘价", "ma10": "MA10(10日均价)", "ma20": "MA20(20日均价)",
                "ma30": "MA30(30日均价)", "lhb_flag": "上龙虎榜",
                "chip_single_peak": "低位单峰密集", "chip_above_avg": "站上成本线",
                "chip_high_diverge": "高位发散", "chip_confidence": "视觉置信度",
                "has_chip": "已有筹码分析", "updated_at": "更新时间",
            }
            cols = ["code", "name", "操作建议", "建议原因", "is_holding", "close",
                    "ma10", "ma20", "ma30", "npr_growth", "pe", "pe_percentile",
                    "lhb_flag", "chip_single_peak", "chip_above_avg", "chip_high_diverge",
                    "chip_confidence", "has_chip", "updated_at"]
            cols = [c for c in cols if c in show.columns or c in ("操作建议", "建议原因")]
            show = show[cols].rename(columns=rename)
            st.dataframe(show, use_container_width=True, hide_index=True)
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
