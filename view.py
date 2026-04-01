from enum import Enum

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from datasets.schema import BASE_SCHEMA
from predict_sales import evaluation, evaluation_mul, f, preprossing_diy


class Container(object):
    def __init__(self):
        self.sales_df: pd.DataFrame = None
        self.prediction_df: pd.DataFrame = None

    def load_sales_df(self):
        self.sales_df = preprossing_diy()
        self.sales_df = BASE_SCHEMA.validate(self.sales_df)


def init_container() -> Container:
    container = Container()
    container.load_sales_df()
    return container


class TIME_FRAME(Enum):
    MONTHLY = "月次 販売額"
    YEARLY = "年度 販売額"
    YEAR14 = "2014年度 販売額"
    YEAR15 = "2015年度 販売額"
    YEAR21 = "2021年度 販売額"
    YEAR22 = "2022年度 販売額"


def build_time_selectbox() -> str | None:
    options = [
        TIME_FRAME.MONTHLY.value,
        TIME_FRAME.YEARLY.value,
        TIME_FRAME.YEAR14.value,
        TIME_FRAME.YEAR15.value,
        # TIME_FRAME.YEAR21.value,
        TIME_FRAME.YEAR22.value,
    ]
    selected = st.sidebar.selectbox(
        label="販売額",
        options=options,
    )
    return selected


class ITEM_FRAME(Enum):
    SALES = "商品販売額"
    DIY = "ＤＩＹ用具・素材"
    ELECTRO = "電気"
    INTERIOR = "インテリア"
    HOUSEHOLD = "家庭用品・日用品"
    GARDENING = "園芸・エクステリア"
    PETS = "ペット・ペット用品"
    CAR = "カー用品・アウトドア"
    OFFICE = "オフィス・カルチャー"
    OTHERS = "その他"


def build_item_selectbox() -> str | None:
    options = [
        ITEM_FRAME.SALES.value,
        ITEM_FRAME.DIY.value,
        ITEM_FRAME.ELECTRO.value,
        ITEM_FRAME.INTERIOR.value,
        ITEM_FRAME.HOUSEHOLD.value,
        ITEM_FRAME.GARDENING.value,
        ITEM_FRAME.PETS.value,
        ITEM_FRAME.CAR.value,
        ITEM_FRAME.OFFICE.value,
        ITEM_FRAME.OTHERS.value,
    ]
    selected = st.sidebar.selectbox(
        label="カテゴリー別予測",
        options=options,
    )
    return selected


class BI(Enum):
    ITEM_SALES = "集計データ"
    PREDICTION = "予測データ"


def build_bi_selectbox() -> str:
    options = [BI.ITEM_SALES.value, BI.PREDICTION.value]
    selected = st.sidebar.selectbox(
        label="",
        options=options,
    )
    return selected


def build(container: Container):
    df = container.sales_df
    bi = build_bi_selectbox()
    if bi is None:
        return
    elif bi == BI.ITEM_SALES.value:
        st.markdown("# ホームセンター商品別販売額")
        st.markdown("### 経済産業省 商業動態統計調査 / 時系列データ")
        st.markdown(
            "###### https://www.e-stat.go.jp/stat-search/files?stat_infid=000031387998"  # noqa: E501
        )
        # st.markdown("###### 最終更新日時：2023-02-15 13:30")
        st.dataframe(df.reset_index())

        time = build_time_selectbox()
        if time == TIME_FRAME.MONTHLY.value:
            st.markdown("### 月次 販売額")
            fig = go.Figure()
            sales_trace = go.Bar(
                x=df.index,
                y=df.商品販売額,
            )
            fig.add_trace(sales_trace)
            st.plotly_chart(fig, use_container_width=True)
        elif time == TIME_FRAME.YEARLY.value:
            st.markdown("### 年度 販売額")
            fig = go.Figure()
            year_trace = go.Bar(
                x=df.resample("Y")
                .sum()
                .reset_index()
                .iloc[:, :1]
                .to_numpy()
                .reshape(-1),
                y=df.商品販売額,
            )
            fig.add_trace(year_trace)
            st.plotly_chart(fig, use_container_width=True)
        elif time == TIME_FRAME.YEAR14.value:
            st.markdown("### 2014年度 販売額")
            fig = go.Figure()
            year14_trace = go.Bar(
                x=df.index[:12],
                y=df.商品販売額,
            )
            fig.add_trace(year14_trace)
            st.plotly_chart(fig, use_container_width=True)
        elif time == TIME_FRAME.YEAR15.value:
            st.markdown("### 2015年度 販売額")
            fig = go.Figure()
            year15_trace = go.Bar(
                x=df.index[12:24],
                y=df["商品販売額"][12:24],
            )
            fig.add_trace(year15_trace)
            st.plotly_chart(fig, use_container_width=True)
        elif time == TIME_FRAME.YEAR21.value:
            st.markdown("### 2021年度 販売額")
            fig = go.Figure()
            year21_trace = go.Bar(
                x=df.index[84:96],
                y=df["商品販売額"][84:96],
            )
            fig.add_trace(year21_trace)
            st.plotly_chart(fig, use_container_width=True)
        elif time == TIME_FRAME.YEAR22.value:
            st.markdown("### 2022年度 販売額")
            fig = go.Figure()
            year22_trace = go.Bar(
                x=df.index[96:108],
                y=df["商品販売額"][96:108],
            )
            fig.add_trace(year22_trace)
            st.plotly_chart(fig, use_container_width=True)
        else:
            ...

    elif bi == BI.PREDICTION.value:
        item = build_item_selectbox()

        def build_prediction(item):
            st.markdown("### 直近３ヶ月の予測結果")
            mae, mse, pred_sales, true_sales = evaluation(item)
            fig = go.Figure()
            predict_trace = go.Bar(
                # x=df.index[105:108],
                y=pred_sales.reshape(-1),
                name="予測値",
            )
            true_trace = go.Bar(
                # x=df.index[105:108],
                y=true_sales.reshape(-1),
                name="実績",
            )
            diff_trace = go.Bar(
                # x=df.index[105:108],
                y=true_sales.reshape(-1) - pred_sales.reshape(-1),
                name="誤差",
            )
            fig.add_trace(predict_trace)
            fig.add_trace(true_trace)
            fig.add_trace(diff_trace)
            st.plotly_chart(fig, use_container_width=True)
            # st.write(f'平均二乗誤差損失: {mse}')
            st.write(f"平均絶対誤差損失: {mae}")

        if item == ITEM_FRAME.SALES.value:
            st.markdown("### 最新2022年12月の予測結果")
            mae, r2, pred_sales, true_sales, intercept, coefs = evaluation_mul(
                item
            )
            lower, upper, y_hat = f(0.05)
            fig = go.Figure()
            predict_trace = go.Bar(
                x=df.index[107:108], y=pred_sales.reshape(-1), name="予測値"
            )
            true_trace = go.Bar(
                x=df.index[107:108], y=true_sales.reshape(-1), name="実績"
            )
            diff_trace = go.Bar(
                # x=df.index[-3],
                x=df.index[107:108],
                y=true_sales.reshape(-1) - pred_sales.reshape(-1),
                name="誤差",
            )
            fig.add_trace(predict_trace)
            fig.add_trace(true_trace)
            fig.add_trace(diff_trace)
            st.plotly_chart(fig, use_container_width=True)
            st.write("回帰式: ")
            st.write(
                f"商品販売額 = {intercept[0]} + {coefs[0][0]}(DIY用品) + {coefs[0][1]}(インテリア)"  # noqa: E501
            )
            st.write(f"寄与率: {r2}")
            # st.write(f'平均絶対誤差損失: {mae}')
            # st.write(f'信頼率95%の母回帰信頼区間: [{lower}, {upper}]')

        elif item == ITEM_FRAME.DIY.value:
            build_prediction(item)
        elif item == ITEM_FRAME.ELECTRO.value:
            build_prediction(item)
        elif item == ITEM_FRAME.INTERIOR.value:
            build_prediction(item)
        elif item == ITEM_FRAME.HOUSEHOLD.value:
            build_prediction(item)
        elif item == ITEM_FRAME.GARDENING.value:
            build_prediction(item)
        elif item == ITEM_FRAME.PETS.value:
            build_prediction(item)
        elif item == ITEM_FRAME.CAR.value:
            build_prediction(item)
        elif item == ITEM_FRAME.OFFICE.value:
            build_prediction(item)
        elif item == ITEM_FRAME.OTHERS.value:
            build_prediction(item)
    else:
        raise ValueError()


if __name__ == "__main__":
    container = init_container()
    build(container=container)
