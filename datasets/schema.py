from pandera import Check, Column, DataFrameSchema


_BASE_SCHEMA = {
    "商品販売額": Column(int, Check(lambda s: s > 0)),
    "ＤＩＹ用具・素材": Column(int, Check(lambda s: s > 0)),
    "電気": Column(int, Check(lambda s: s > 0)),
    "インテリア": Column(int, Check(lambda s: s > 0)),
    "家庭用品・日用品": Column(int, Check(lambda s: s > 0)),
    "園芸・エクステリア": Column(int, Check(lambda s: s > 0)),
    "ペット・ペット用品": Column(int, Check(lambda s: s > 0)),
    "カー用品・アウトドア": Column(int, Check(lambda s: s > 0)),
    "オフィス・カルチャー": Column(int, Check(lambda s: s > 0)),
    "その他": Column(int, Check(lambda s: s > 0)),
}
BASE_SCHEMA = DataFrameSchema(
    _BASE_SCHEMA,
)
