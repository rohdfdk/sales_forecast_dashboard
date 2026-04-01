import numpy as np
import pandas as pd
import scipy as sp
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from torch import nn

xls = pd.ExcelFile("./datasets/h2slt44j.xls")
df = pd.read_excel(xls, sheet_name="販売額(value)月次(Monthly)", skiprows=6)


def preprossing_diy():
    def tweak_diy(df):
        return (
            df.drop(
                columns=["時間軸コード", "Month", "Year", "Number of establishments"]
            )
            .rename(
                columns={
                    "年月": "Date",
                    "sales of goods": "商品販売額",
                    "D.I.Y. tools and materials ": "ＤＩＹ用具・素材",
                    "Gardening and exteriors ": "園芸・エクステリア",
                    "Electric appliances": "電気",
                    "Household utensils and daily necessities": "家庭用品・日用品",
                    "Pet and pet products": "ペット・ペット用品",
                    "Car supplies and outdoor goods": "カー用品・アウトドア",
                    "Office products and hobbies": "オフィス・カルチャー",
                    "Interiors": "インテリア",
                    "Others": "その他",
                }
            )
            .astype({"Date": "string"})
            .assign(
                Date=lambda _df: (
                    pd.to_datetime(
                        _df["Date"].str.replace("年", "/").str.replace("月", ""),
                        format="ISO8601",
                    )
                )
            )
            .set_index(["Date"])
        )

    diy = tweak_diy(df)
    return diy


def define_nn(item):
    pth = item + ".pth"
    in_features = 1
    k = 200
    out_features = 1

    class Net(nn.Module):
        def __init__(self, in_features, k, out_features):
            super(Net, self).__init__()
            self.linear1 = torch.nn.Linear(in_features, k)
            self.relu = torch.nn.ReLU()
            self.linear2 = torch.nn.Linear(k, out_features)

        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(in_features, k, out_features).to(device)
    params = torch.load(f"./datasets/{pth}", map_location=device)
    net.load_state_dict(params)

    return net, device


def predict_sales(net, device, X_true, y_true):
    net.eval()
    pred_sales = []
    true_sales = []
    with torch.no_grad():
        xx = X_true.to(device)
        yy = y_true.to(device)
        y_pred_proba = net(xx)
        pred_sales.append(y_pred_proba.view(-1).tolist())
        true_sales.append(yy.view(-1).tolist())
    pred_sales = [e for _ in pred_sales for e in _]
    true_sales = [e for _ in true_sales for e in _]
    pred_sales = torch.tensor(pred_sales).reshape(-1, 1)
    true_sales = torch.tensor(true_sales).reshape(-1, 1)
    mae = mean_absolute_error(true_sales, pred_sales)
    mse = ((pred_sales.reshape(-1) - true_sales.reshape(-1)) ** 2).mean()

    return mae, mse, pred_sales, true_sales


def evaluation_mul(item):
    diy = preprossing_diy()
    X_train = diy[["ＤＩＹ用具・素材", "インテリア"]].loc["2015-01":"2022-06"]
    y_train = diy[[item]].loc["2015-01":"2022-06"]
    # X_test = diy[['ＤＩＹ用具・素材', 'インテリア']].loc['2022-07':'2022-09']
    # y_test = diy[[item]].loc['2022-07':'2022-09']
    X_true = diy[["ＤＩＹ用具・素材", "インテリア"]].loc["2022-10":"2022-12"]
    y_true = diy[[item]].loc["2022-10":"2022-12"]
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=True, random_state=123
    )
    lr = LinearRegression().fit(X_tr, y_tr)
    pred_sales = (
        lr.intercept_
        + lr.coef_[0][0] * diy["ＤＩＹ用具・素材"].loc["2022-12"]
        + lr.coef_[0][1] * diy["インテリア"].loc["2022-12"]
    ).to_numpy()
    true_sales = torch.Tensor(diy[[item]].loc["2022-12"].to_numpy())
    mae = mean_absolute_error(pred_sales, true_sales)
    r2 = lr.score(X_true, y_true)
    # intercept = lr.intercept_
    # coefs = lr.coef_
    return mae, r2, pred_sales, true_sales, lr.intercept_, lr.coef_


def evaluation(item):
    diy = preprossing_diy()
    if item == "商品販売額":
        evaluation_mul(item)

    if item == "ＤＩＹ用具・素材":
        net, device = define_nn(item)
        X_true = torch.Tensor(diy.iloc[105:108, 1:2].to_numpy())
    elif item == "電気":
        net, device = define_nn(item)
        X_true = torch.Tensor(diy.iloc[105:108, 2:3].to_numpy())
    elif item == "インテリア":
        net, device = define_nn(item)
        X_true = torch.Tensor(diy.iloc[105:108, 3:4].to_numpy())
    elif item == "家庭用品・日用品":
        net, device = define_nn(item)
        X_true = torch.Tensor(diy.iloc[105:108, 1:2].to_numpy())
    elif item == "園芸・エクステリア":
        net, device = define_nn(item)
        X_true = torch.Tensor(diy.iloc[105:108, 1:2].to_numpy())
    elif item == "ペット・ペット用品":
        net, device = define_nn(item)
        X_true = torch.Tensor(diy.iloc[105:108, 1:2].to_numpy())
    elif item == "カー用品・アウトドア":
        net, device = define_nn(item)
        X_true = torch.Tensor(diy.iloc[105:108, 1:2].to_numpy())
    elif item == "オフィス・カルチャー":
        net, device = define_nn(item)
        X_true = torch.Tensor(diy.iloc[105:108, 1:2].to_numpy())
    elif item == "その他":
        net, device = define_nn(item)
        X_true = torch.Tensor(diy.iloc[105:108, 1:2].to_numpy())
    else:
        ...
    y_true = torch.Tensor(diy[[item]].loc["2022-10":"2022-12"].to_numpy())
    mae, mse, pred_sales, true_sales = predict_sales(
        net, device, X_true, y_true
    )
    return mae, mse, pred_sales, true_sales


def f(alpha):
    diy = preprossing_diy()
    x1 = 71675
    x2 = 23695
    beta_0 = 19584.52379184
    beta_1 = 2.50815121
    beta_2 = 5.34599449
    n = len(diy.index) - 1
    p = 2
    phi_e = n - 1 - p
    my = diy["商品販売額"].loc["2015-01":"2022-09"].mean()
    mx1 = diy["ＤＩＹ用具・素材"].loc["2015-01":"2022-09"].mean()
    mx2 = diy["インテリア"].loc["2015-01":"2022-09"].mean()
    s11 = ((diy["ＤＩＹ用具・素材"].loc["2015-01":"2022-09"] - mx1) ** 2).sum()
    s22 = ((diy["インテリア"].loc["2015-01":"2022-09"] - mx2) ** 2).sum()
    s12 = (
        (diy["ＤＩＹ用具・素材"].loc["2015-01":"2022-09"] - mx1)
        * (diy["インテリア"].loc["2015-01":"2022-09"] - mx2)
    ).sum()
    syy = ((diy["商品販売額"].loc["2015-01":"2022-09"] - my) ** 2).sum()
    s1y = (
        (diy["ＤＩＹ用具・素材"].loc["2015-01":"2022-09"] - mx1)
        * (diy["商品販売額"].loc["2015-01":"2022-09"] - my)
    ).sum()
    s2y = (
        (diy["インテリア"].loc["2015-01":"2022-09"] - mx2)
        * (diy["商品販売額"].loc["2015-01":"2022-09"] - my)
    ).sum()
    sr = beta_1 * s1y + beta_2 * s2y
    se = syy - sr
    ve = se / phi_e
    mat = np.array([[s11, s12], [s12, s22]])
    inv_mat = sp.linalg.inv(mat)
    s11_inv = inv_mat[0][0]
    s12_inv = inv_mat[0][1]
    s22_inv = inv_mat[1][1]
    d = (n - 1) * (
        (((x1 - mx1) ** 2) * s11_inv)
        + (2 * (x1 - mx1) * (x2 - mx2) * s12_inv)
        + (((x2 - mx2) ** 2) * s22_inv)
    )
    lev1 = 1 + (1 / n) + (d / n - 1)
    rv = sp.stats.t(phi_e)
    interval = rv.isf(alpha / 2) * np.sqrt(lev1 * ve)
    y_hat = beta_0 + beta_1 * x1 + beta_2 * x2
    lower = y_hat - interval
    upper = y_hat + interval
    # r22 = sr/syy

    return lower, upper, y_hat
