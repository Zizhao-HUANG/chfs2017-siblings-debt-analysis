import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import warnings

# Suppress specific warnings if needed
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning) # Ignore potential sklearn warnings

# --- Configuration ---
hh_file_path = '/Users/lyahnw/Downloads/chfs2017_hh_202206.dta'
ind_file_path = '/Users/lyahnw/Downloads/chfs2017_ind_202206.dta'
# Updated output filename to reflect new models
output_file_path = '/Users/lyahnw/Downloads/chfs2017_processed_siblings_debt_v7_multi_model.csv'

# --- Check if files exist ---
if not os.path.exists(hh_file_path):
    print(f"错误：家庭文件未在 {hh_file_path} 找到")
    exit()
if not os.path.exists(ind_file_path):
    print(f"错误：个人文件未在 {ind_file_path} 找到")
    exit()

print("正在加载数据...")
# --- Load Data ---
try:
    hh_df = pd.read_stata(hh_file_path, convert_categoricals=False)
    print(f"家庭数据已加载，包含 {len(hh_df)} 行， {len(hh_df.columns)} 列。")
    ind_df = pd.read_stata(ind_file_path, convert_categoricals=False)
    print(f"个人数据已加载，包含 {len(ind_df)} 行， {len(ind_df.columns)} 列。")
    print("数据加载成功。")
except Exception as e:
    print(f"加载 Stata 文件时出错: {e}")
    exit()

# --- Process Individual Data (ind_df) ---
print("正在处理个人数据以查找户主（问卷回答者代理）信息...")
head_relation_var = 'a2001'
if head_relation_var not in ind_df.columns:
     print(f"错误: 无法在个人数据中找到用于识别户主代理的变量 '{head_relation_var}'。")
     exit()
heads_df = ind_df[ind_df[head_relation_var] == 1].copy()
print(f"识别出 {len(heads_df)} 位户主代理（问卷回答者）。")

# Calculate Head's Age
if 'a2005' not in heads_df.columns:
    print("错误：个人数据中缺少 'a2005' (出生年份) 列。")
    exit()
heads_df['head_age'] = 2017 - heads_df['a2005']

# --- Add Age Filter for Head (>=16) ---
# Filter out heads younger than 16 (as per questionnaire logic for respondent)
initial_head_count = len(heads_df)
heads_df = heads_df[heads_df['head_age'] >= 16].copy()
removed_count = initial_head_count - len(heads_df)
if removed_count > 0:
    print(f"已应用年龄筛选 (head_age >= 16)，排除了 {removed_count} 位年龄异常的户主代理。")


# Calculate Head's Total Siblings (A2028: brothers, A2029: sisters)
# Questionnaire Q69, Q70 state these are asked only for respondent/spouse aged 40 or below
if 'a2028' not in heads_df.columns: heads_df['a2028'] = 0
if 'a2029' not in heads_df.columns: heads_df['a2029'] = 0
heads_df['head_siblings_raw'] = heads_df['a2028'].fillna(0) + heads_df['a2029'].fillna(0)
heads_df['head_siblings'] = np.where(heads_df['head_age'] <= 40, heads_df['head_siblings_raw'], np.nan)

# Select relevant head columns to merge
head_control_cols_map = {
    'hhid': 'hhid',
    'head_age': 'head_age',
    'a2003': 'head_sex',       # Q38/A1113: Gender (1=Male, 2=Female)
    'a2012': 'head_educ',      # Q43: Education level
    'a2024': 'head_marital',   # Q67: Marital status
    'a2025b': 'head_health',   # Q68: Health status compared to peers (1=Very Good, 5=Very Poor)
    'head_siblings': 'head_siblings' # Calculated above based on A2028, A2029, A2005
}
head_cols_to_select = [col for col in head_control_cols_map.keys() if col in heads_df.columns]
final_head_cols_map = {k: v for k, v in head_control_cols_map.items() if k in head_cols_to_select}
head_info_to_merge = heads_df[head_cols_to_select].rename(columns=final_head_cols_map)
head_info_to_merge = head_info_to_merge.drop_duplicates(subset=['hhid'], keep='first')
print(f"准备合并 {len(head_info_to_merge)} 个家庭的户主代理信息。")

# --- Merge Head Info into Household Data (hh_df) ---
print("正在将户主代理信息合并到家庭数据中...")
original_hh_rows = len(hh_df)
hh_df = pd.merge(hh_df, head_info_to_merge, on='hhid', how='left')
if len(hh_df) != original_hh_rows:
    print(f"警告：合并后行数发生变化（从 {original_hh_rows} 到 {len(hh_df)}），请检查合并键 'hhid' 的唯一性。")


# --- Define Asset and Debt Components (Variable names as provided before) ---
print("正在定义资产和负债变量列表...")
# (Variable lists remain the same as before - using the corrected ones from previous step)
# --- Debt Variables ---
debt_bus_bank = ['b3005b_2']; debt_bus_private = ['b3031a_2']; debt_bus_private_it = ['b3031ait_2']
debt_house_bank = [f'c2064_{i}' for i in range(1, 7)]; debt_house_bank_it = [f'c2064it_{i}' for i in range(1, 7)]
debt_house_other = [f'c3002a_{i}' for i in range(1, 7)]; debt_house_other_it = [f'c3002ait_{i}' for i in range(1, 7)]
debt_house_agg_other = ['c2023e']; debt_house_agg_other_it = ['c2023eit']
debt_house_collateral = ['c3017ca']; debt_house_collateral_it = ['c3017cait']
debt_shop_bank = ['c3019c']; debt_shop_bank_it = ['c3019cit']; debt_shop_other = ['c3019e']; debt_shop_other_it = ['c3019eit']
debt_car = ['c7060']; debt_car_it = ['c7060it']; debt_vehicle_other = ['c7061']; debt_vehicle_other_it = ['c7061it']
debt_durable = ['c8007']; debt_durable_it = ['c8007it']
debt_stock = ['d3116b']; debt_finance_other = ['d9108']; debt_finance_other_it = ['d9108it']
debt_edu_bank = ['e1006']; debt_edu_bank_it = ['e1006it']; debt_edu_private = ['e1022']; debt_edu_private_it = ['e1022it']
debt_medical = ['e4003']; debt_medical_it = ['e4003it']; debt_other = ['e3003c']; debt_other_it = ['e3003cit']
all_debt_exact_vars = (debt_bus_bank + debt_bus_private + debt_house_bank + debt_house_other + debt_house_agg_other + debt_house_collateral + debt_shop_bank + debt_shop_other + debt_car + debt_vehicle_other + debt_durable + debt_stock + debt_finance_other + debt_edu_bank + debt_edu_private + debt_medical + debt_other)
all_debt_interval_vars = (debt_bus_private_it + debt_house_bank_it + debt_house_other_it + debt_house_agg_other_it + debt_house_collateral_it + debt_shop_bank_it + debt_shop_other_it + debt_car_it + debt_vehicle_other_it + debt_durable_it + debt_finance_other_it + debt_edu_bank_it + debt_edu_private_it + debt_medical_it + debt_other_it)

# --- Asset Variables ---
asset_bus = ['b2003d']; asset_bus_it = ['b2003dit']
asset_house = [f'c2016_{i}' for i in range(1, 7)]; asset_house_it = [f'c2016it_{i}' for i in range(1, 7)]
asset_house_agg_other = ['c2023d']; asset_house_agg_other_it = ['c2023dit']
asset_shop = ['c3019a']; asset_shop_it = ['c3019ait']
asset_car = ['c7052b']; asset_car_it = ['c7052bit']; asset_vehicle_comm = ['c7059']; asset_vehicle_other = ['c7058']
vehicle_in_business_col = 'c7062'; vehicle_in_business_col_it = 'c7062it'
asset_durable = ['c8002']; asset_other_nonfin = ['c8005']
asset_deposit_checking = ['d1105']; asset_deposit_checking_it = ['d1105it']; asset_deposit_savings = ['d2104']; asset_deposit_savings_it = ['d2104it']
asset_stock_cash = ['d3103']; asset_stock_cash_it = ['d3103it']; asset_stock_value = ['d3109']; asset_stock_value_it = ['d3109it']; asset_stock_nonpublic = ['d3116']; asset_stock_nonpublic_it = ['d3116it']
asset_fund = ['d5107']; asset_fund_it = ['d5107it']; asset_internet_finance = ['d7106h']; asset_internet_finance_it = ['d7106hit']; asset_other_finance_prod = ['d7110a']; asset_other_finance_prod_it = ['d7110ait']
asset_bond = [f'd4103_{i}' for i in range(1, 6)]; asset_bond_it = [f'd4103it_{i}' for i in range(1, 6)]
asset_derivative = ['d6100a']; asset_derivative_it = ['d6100ait']; asset_non_rmb = ['d8104']; asset_non_rmb_it = ['d8104it']; asset_gold = ['d9103']; asset_gold_it = ['d9103it']; asset_other_fin = ['d9110a']; asset_other_fin_it = ['d9110ait']
asset_cash = ['k1101']; asset_cash_it = ['k1101it']; asset_receivable = ['k2102c']; asset_receivable_it = ['k2102cit']
all_asset_exact_vars = (asset_bus + asset_house + asset_house_agg_other + asset_shop + asset_car + asset_vehicle_comm + asset_vehicle_other + asset_durable + asset_other_nonfin + asset_deposit_checking + asset_deposit_savings + asset_stock_cash + asset_stock_value + asset_stock_nonpublic + asset_fund + asset_internet_finance + asset_other_finance_prod + asset_bond + asset_derivative + asset_non_rmb + asset_gold + asset_other_fin + asset_cash + asset_receivable)
all_asset_interval_vars = (asset_bus_it + asset_house_it + asset_house_agg_other_it + asset_shop_it + asset_car_it + asset_deposit_checking_it + asset_deposit_savings_it + asset_stock_cash_it + asset_stock_value_it + asset_stock_nonpublic_it + asset_fund_it + asset_internet_finance_it + asset_other_finance_prod_it + asset_bond_it + asset_derivative_it + asset_non_rmb_it + asset_gold_it + asset_other_fin_it + asset_cash_it + asset_receivable_it)


# --- Helper Function to Calculate Midpoints (Corrected based on Questionnaire) ---
def get_midpoint(interval_code, var_name):
    """
    根据 CHFS 2017 问卷定义，计算区间变量的中点值（单位：元 或 平方米）。
    对“X以上”的开放区间，使用下限的1.5倍作为估计。

    Args:
        interval_code (float/int): 问卷中记录的区间编码。
        var_name (str): 区间变量的名称 (e.g., 'b2003ait', 'c2016it_1')。

    Returns:
        float: 计算出的中点值，如果无法匹配则返回 np.nan。
    """
    if pd.isna(interval_code):
        return np.nan

    # Remove suffix like _1, _2 for general mapping if needed (depends on how vars are listed)
    base_var_name = var_name.split('_')[0] if '_' in var_name and var_name[-2]=='_' else var_name # Handle c2016it_1 etc.

    # --- Define Mappings based on Questionnaire ---
    # Note: Upper bound estimation uses 1.5 * lower bound

    # Mapping 1: Q158, Q172, Q181, Q184, Q187, Q565, Q567, Q614, Q587, Q589, Q618, Q623, Q627, Q631, Q649, Q460
    map1 = {1: 5000, 2: 20000, 3: 40000, 4: 60000, 5: 85000, 6: 200000, 7: 400000, 8: 750000, 9: 3000000, 10: 7500000, 11: 15000000}
    vars_map1 = ['b2003ait', 'b2050it', 'b2059it', 'b2063it', 'b2080it', 'd3109it', 'd3110it', 'd4103it',
                 'd5107it', 'd5108it', 'd6100ait', 'd8104it', 'd9103it', 'd9110ait', 'k2102cit', 'c3019ait']

    # Mapping 2: Q162, Q175 (Same as Map 1)
    map2 = map1
    vars_map2 = ['b2003bit', 'b2052it']

    # Mapping 3: Q164, Q178, Q120, Q579, Q170, Q213, Q215, Q217, Q219, Q238, Q240, Q242, Q244, Q261, Q452, Q463, Q466, Q481, Q483, Q513, Q515, Q541, Q546, Q549, Q560, Q599, Q607, Q663, Q670, Q698, Q678, Q851, Q351
    map3 = {1: 5000, 2: 15000, 3: 35000, 4: 75000, 5: 150000, 6: 250000, 7: 400000, 8: 750000, 9: 1500000, 10: 3500000, 11: 7500000}
    vars_map3 = ['b2003eit', 'b2055it', 'a3136it', 'd3117it', 'b2046it', 'b3004bit', 'b3005bit', 'b3005it', 'b3006ait',
                 'b3030dit', 'b3030eit', 'b3031ait', 'b3045cit', 'b3056ait', 'c3017cait', 'c3019cit', 'c3019eit',
                 'c7052bit', 'c7060it', 'c7061it', 'c7062it', 'c8007it', 'd1105it', 'd2104it', 'd3103it', 'd7106hit',
                 'd7110ait', 'e1006it', 'e1022it', 'e3003cit', 'e4003it', 'h2004it', 'c2035ait']

    # Mapping 4: Q190, Q122, Q124, Q126, Q205, Q595, Q601, Q609, Q629, Q637, Q633, Q646, Q660, Q716, Q722
    map4 = {1: 2500, 2: 7500, 3: 15000, 4: 35000, 5: 75000, 6: 125000, 7: 175000, 8: 250000, 9: 400000, 10: 750000, 11: 1500000}
    vars_map4 = ['b2093it', 'a3136ait', 'a3136bit', 'a3137it', 'b2110it', 'd5109it', 'd7106jit', 'd7112it',
                 'd9105it', 'd9108it', 'd9110bit', 'k1101it', 'k2208it', 'f1010it', 'f1031it']

    # Mapping 5: Q230
    map5 = {1: 500, 2: 1500, 3: 3500, 4: 7500, 5: 15000, 6: 35000, 7: 75000, 8: 150000}
    vars_map5 = ['b3008fit']

    # Mapping 6: Q282 (Unit: sqm)
    map6 = {1: 25, 2: 60.5, 3: 80.5, 4: 95.5, 5: 110.5, 6: 132, 7: 172, 8: 300}
    vars_map6 = ['c1000bbit']

    # Mapping 7: Q284
    map7 = {1: 50000, 2: 200000, 3: 400000, 4: 600000, 5: 850000, 6: 1250000, 7: 2250000, 8: 4000000, 9: 6000000, 10: 8500000, 11: 12500000, 12: 17500000, 13: 30000000}
    vars_map7 = ['c1000bdit']

    # Mapping 8: Q320
    map8 = {1: 5000, 2: 15000, 3: 35000, 4: 75000, 5: 150000, 6: 250000, 7: 400000, 8: 750000, 9: 1500000, 10: 3500000, 11: 6000000, 12: 8500000, 13: 12500000, 14: 17500000, 15: 30000000}
    vars_map8 = ['c2000fit']

    # Mapping 9: Q335, Q338
    map9 = {1: 5000, 2: 20000, 3: 40000, 4: 60000, 5: 85000, 6: 200000, 7: 400000, 8: 750000, 9: 3000000, 10: 7500000, 11: 12500000, 12: 17500000, 13: 30000000}
    vars_map9 = ['c2013it', 'c2016it']

    # Mapping 10: Q344, Q347, Q361
    map10 = {1: 50000, 2: 150000, 3: 350000, 4: 650000, 5: 900000, 6: 1250000, 7: 1750000, 8: 3500000, 9: 6500000, 10: 9000000, 11: 15000000}
    vars_map10 = ['c2027dit', 'c2032it', 'c2064it']

    # Mapping 11: Q355
    map11 = {1: 500, 2: 2000, 3: 4000, 4: 6500, 5: 9000, 6: 12500, 7: 17500, 8: 25000, 9: 40000, 10: 75000}
    vars_map11 = ['c2045it']

    # Mapping 12: Q364, Q366
    map12 = {1: 25000, 2: 75000, 3: 150000, 4: 250000, 5: 400000, 6: 650000, 7: 900000, 8: 1250000, 9: 1750000, 10: 3500000, 11: 7500000}
    vars_map12 = ['c3002it', 'c3002ait']

    # Mapping 13: Q468, Q470, Q616, Q620
    map13 = {1: 5000, 2: 15000, 3: 35000, 4: 75000, 5: 150000, 6: 250000, 7: 400000, 8: 750000, 9: 1500000, 10: 3500000, 11: 7500000}
    vars_map13 = ['c3024it', 'c3025it', 'd4111it', 'd6116it']

    # Mapping 14: Q534, Q809, Q811, Q813, Q815, Q817, Q538, Q728, Q744
    map14 = {1: 1000, 2: 3500, 3: 7500, 4: 15000, 5: 35000, 6: 75000, 7: 125000, 8: 175000, 9: 250000, 10: 400000, 11: 750000}
    vars_map14 = ['c8002ait', 'g1017it', 'g1018it', 'g1019it', 'g1019ait', 'g1020it', 'c8005ait', 'f2006it', 'f4011it']

    # Mapping 15: Q625
    map15 = {1: 10000, 2: 35000, 3: 75000, 4: 150000, 5: 350000, 6: 750000, 7: 1500000, 8: 3500000, 9: 7500000, 10: 15000000, 11: 30000000}
    vars_map15 = ['d8106it']

    # Mapping 16: Q706
    map16 = {1: 5000, 2: 15000, 3: 35000, 4: 75000, 5: 150000, 6: 250000, 7: 400000, 8: 750000, 9: 1500000, 10: 3500000, 11: 7500000}
    vars_map16 = ['e3005cit']

    # Mapping 17: Q713
    map17 = {1: 25, 2: 75, 3: 125, 4: 225, 5: 400, 6: 650, 7: 1150, 8: 2250, 9: 4000, 10: 7500, 11: 15000, 12: 25000, 13: 40000, 14: 75000}
    vars_map17 = ['f1005it']

    # Mapping 18: Q737
    map18 = {1: 100, 2: 250, 3: 400, # Option 4 (500以下) might be missing in some waves
             5: 750, 6: 1500, 7: 2500, 8: 4000, 9: 6500, 10: 11500, 11: 17500, 12: 30000}
    vars_map18 = ['f4005it']

    # Mapping 19: Q740
    map19 = {1: 500, 2: 2000, 3: 4000, # Option 4 (5千以下) might be missing
             5: 7500, 6: 15000, 7: 35000, 8: 75000, 9: 125000, 10: 175000, 11: 250000, 12: 400000, 13: 750000, 14: 1500000}
    vars_map19 = ['f4008it']

    # Mapping 20: Q902
    map20 = {1: 250, 2: 750, 3: 1500, 4: 3500, 5: 7500, 6: 15000, 7: 30000}
    vars_map20 = ['h3351it']

    # Mapping 21: Q906, Q909
    map21 = {1: 250, 2: 750, 3: 2000, 4: 4000, 5: 7500, 6: 15000, 7: 35000, 8: 75000}
    vars_map21 = ['h3354it', 'h3356it']

    # Mapping 22: Q918, Q920, Q922
    map22 = {1: 50, 2: 300, 3: 750, 4: 3000, 5: 7500, 6: 30000, 7: 75000}
    vars_map22 = ['h3367it', 'h3368it', 'h3369it']

    # Mapping 23: Q826
    map23 = {1: 150, 2: 450, 3: 800, 4: 1250, 5: 2250, 6: 4500, 7: 8000, 8: 15000, 9: 35000, 10: 75000, 11: 150000}
    vars_map23 = ['g1024it']

    # --- Apply Mapping ---
    if base_var_name in vars_map1: return map1.get(interval_code, np.nan)
    if base_var_name in vars_map2: return map2.get(interval_code, np.nan)
    if base_var_name in vars_map3: return map3.get(interval_code, np.nan)
    if base_var_name in vars_map4: return map4.get(interval_code, np.nan)
    if base_var_name in vars_map5: return map5.get(interval_code, np.nan)
    if base_var_name in vars_map6: return map6.get(interval_code, np.nan) # Note: Unit is sqm
    if base_var_name in vars_map7: return map7.get(interval_code, np.nan)
    if base_var_name in vars_map8: return map8.get(interval_code, np.nan)
    if base_var_name in vars_map9: return map9.get(interval_code, np.nan)
    if base_var_name in vars_map10: return map10.get(interval_code, np.nan)
    if base_var_name in vars_map11: return map11.get(interval_code, np.nan)
    if base_var_name in vars_map12: return map12.get(interval_code, np.nan)
    if base_var_name in vars_map13: return map13.get(interval_code, np.nan)
    if base_var_name in vars_map14: return map14.get(interval_code, np.nan)
    if base_var_name in vars_map15: return map15.get(interval_code, np.nan)
    if base_var_name in vars_map16: return map16.get(interval_code, np.nan)
    if base_var_name in vars_map17: return map17.get(interval_code, np.nan)
    if base_var_name in vars_map18: return map18.get(interval_code, np.nan)
    if base_var_name in vars_map19: return map19.get(interval_code, np.nan)
    if base_var_name in vars_map20: return map20.get(interval_code, np.nan)
    if base_var_name in vars_map21: return map21.get(interval_code, np.nan)
    if base_var_name in vars_map22: return map22.get(interval_code, np.nan)
    if base_var_name in vars_map23: return map23.get(interval_code, np.nan)

    # Fallback or warning if no mapping found
    # print(f"警告: 未找到变量 '{var_name}' (基础变量 '{base_var_name}') 的特定中点映射。返回 NaN。") # Keep this commented unless debugging
    return np.nan


# --- Data Cleaning and Calculation ---
print("正在计算总资产、总负债和负债率...")
print("注意：区间变量将使用基于 CHFS 2017 问卷估计的中点值。")

# Ensure all listed columns exist, create as NaN if missing
all_vars_needed = list(set(all_debt_exact_vars + all_debt_interval_vars +
                           all_asset_exact_vars + all_asset_interval_vars +
                           [vehicle_in_business_col, vehicle_in_business_col_it]))
missing_cols_count = 0
for col in all_vars_needed:
    if col not in hh_df.columns:
        hh_df[col] = np.nan
        missing_cols_count += 1
if missing_cols_count > 0: print(f"警告：共创建了 {missing_cols_count} 个缺失的列作为 NaN。")

# --- Coalesce Exact and Interval Values ---
print("正在合并精确值和区间中点值...")
coalesced_debt_vars = []
for i in range(len(all_debt_exact_vars)):
    exact = all_debt_exact_vars[i]
    interval = all_debt_interval_vars[i] if i < len(all_debt_interval_vars) else None
    val_col = f"{exact}_val"

    if exact not in hh_df:
        hh_df[val_col] = np.nan
        continue

    if interval and interval in hh_df:
        midpoint_col = hh_df[interval].apply(lambda x: get_midpoint(x, var_name=interval))
        hh_df[val_col] = hh_df[exact].combine_first(midpoint_col)
    else:
         hh_df[val_col] = hh_df[exact]

    coalesced_debt_vars.append(val_col)


coalesced_asset_vars = []
for i in range(len(all_asset_exact_vars)):
     exact = all_asset_exact_vars[i]
     interval = all_asset_interval_vars[i] if i < len(all_asset_interval_vars) else None
     val_col = f"{exact}_val"

     if exact not in hh_df:
         hh_df[val_col] = np.nan
         continue

     if interval and interval in hh_df:
         midpoint_col = hh_df[interval].apply(lambda x: get_midpoint(x, var_name=interval))
         hh_df[val_col] = hh_df[exact].combine_first(midpoint_col)
     else:
         hh_df[val_col] = hh_df[exact]

     coalesced_asset_vars.append(val_col)

# Handle vehicle adjustment variable separately
vehicle_adj_val_col = f"{vehicle_in_business_col}_val"
if vehicle_in_business_col in hh_df and vehicle_in_business_col_it in hh_df:
    midpoint_veh_adj = hh_df[vehicle_in_business_col_it].apply(lambda x: get_midpoint(x, var_name=vehicle_in_business_col_it))
    hh_df[vehicle_adj_val_col] = hh_df[vehicle_in_business_col].combine_first(midpoint_veh_adj)
elif vehicle_in_business_col in hh_df:
     hh_df[vehicle_adj_val_col] = hh_df[vehicle_in_business_col]
else:
     hh_df[vehicle_adj_val_col] = 0

# --- Calculate Totals using Coalesced Values ---
valid_coalesced_debt_vars = [col for col in coalesced_debt_vars if col in hh_df.columns]
valid_coalesced_asset_vars = [col for col in coalesced_asset_vars if col in hh_df.columns]

hh_df['total_debt'] = hh_df[valid_coalesced_debt_vars].fillna(0).sum(axis=1)
hh_df['total_assets_raw'] = hh_df[valid_coalesced_asset_vars].fillna(0).sum(axis=1)

vehicle_adjustment = hh_df[vehicle_adj_val_col].fillna(0) if vehicle_adj_val_col in hh_df else 0
hh_df['total_assets'] = hh_df['total_assets_raw'] - vehicle_adjustment
hh_df['total_assets'] = hh_df['total_assets'].apply(lambda x: max(x, 0))
print("总负债和总资产计算完成。")

# --- Calculate Debt Ratio ---
print("正在计算负债率...")
epsilon = 1e-9 # Use a smaller epsilon
hh_df['debt_ratio'] = hh_df['total_debt'] / (hh_df['total_assets'] + epsilon)
hh_df.loc[(hh_df['total_debt'] == 0) & (hh_df['total_assets'] == 0), 'debt_ratio'] = 0
hh_df.loc[(hh_df['total_debt'] > 0) & (hh_df['total_assets'] == 0), 'debt_ratio'] = np.nan # Assign NaN for infinite ratio
print("负债率计算完成。")


# --- Winsorize Debt Ratio ---
debt_ratio_clean = hh_df['debt_ratio'].replace([np.inf, -np.inf], np.nan).dropna()
if not debt_ratio_clean.empty:
    winsorized_values = winsorize(debt_ratio_clean, limits=[0.01, 0.01])
    hh_df['debt_ratio_winsorized'] = pd.Series(winsorized_values, index=debt_ratio_clean.index)
    print(f"已对 debt_ratio 进行 Winsorize 处理 (1% 上下限)。新列为 'debt_ratio_winsorized'。")
else:
    print("警告：debt_ratio 列清理后为空，无法进行 Winsorize 处理。")
    hh_df['debt_ratio_winsorized'] = np.nan


# --- Prepare Control Variables ---
print("正在准备控制变量...")
if 'head_sex' in hh_df.columns:
    hh_df['head_is_male'] = np.where(hh_df['head_sex'] == 1, 1, 0)
    hh_df.loc[hh_df['head_sex'].isna(), 'head_is_male'] = np.nan
else:
    hh_df['head_is_male'] = np.nan
    print("警告：'head_sex' 列缺失，无法创建 'head_is_male'。")

if 'head_marital' in hh_df.columns:
    hh_df['head_is_married'] = np.where(hh_df['head_marital'].isin([2, 3, 7]), 1, 0)
    hh_df.loc[hh_df['head_marital'].isna(), 'head_is_married'] = np.nan
else:
     hh_df['head_is_married'] = np.nan
     print("警告：'head_marital' 列缺失，无法创建 'head_is_married'。")

if 'b2000b' in hh_df.columns:
    hh_df['has_business'] = np.where(hh_df['b2000b'] == 1, 1, 0)
    hh_df.loc[hh_df['b2000b'].isna(), 'has_business'] = np.nan
else:
    hh_df['has_business'] = np.nan
    print("警告：'b2000b' 列缺失，无法创建 'has_business'。")

if 'c2002' in hh_df.columns:
    # Fill NaN with 0 for num_houses, assuming NaN means no houses owned or info missing
    hh_df['num_houses'] = hh_df['c2002'].fillna(0)
else:
    hh_df['num_houses'] = 0 # Assume 0 if column missing
    print("警告：'c2002' 列缺失，'num_houses' 设为 0。")

# --- Log Transform Total Assets ---
print("正在对 total_assets 进行对数变换...")
hh_df['log_total_assets'] = np.log(hh_df['total_assets'] + 1) # Add 1 to handle zero assets

# --- Create Log Transformed Dependent Variable (for robustness check) ---
print("正在创建对数变换后的负债率（用于稳健性检验）...")
small_constant_dv = 0.001
if 'debt_ratio_winsorized' in hh_df.columns:
    # Ensure the input to log is positive
    log_input = hh_df['debt_ratio_winsorized'] + small_constant_dv
    hh_df['log_debt_ratio_winsorized'] = np.log(log_input.where(log_input > 0)) # Apply log only where input > 0
else:
    hh_df['log_debt_ratio_winsorized'] = np.nan
    print("警告：'debt_ratio_winsorized' 列缺失，无法创建对数变换后的负债率。")


# --- Select Final Columns for Analysis ---
print("正在选择最终分析所需的列...")
core_vars = ['hhid', 'head_siblings', 'debt_ratio_winsorized', 'log_debt_ratio_winsorized', 'total_debt', 'total_assets']
head_controls = ['head_age', 'head_is_male', 'head_educ', 'head_is_married', 'head_health']
hh_controls = ['has_business', 'num_houses', 'log_total_assets']
final_analysis_cols = core_vars + head_controls + hh_controls

existing_final_cols = [col for col in final_analysis_cols if col in hh_df.columns]
missing_final_cols = [col for col in final_analysis_cols if col not in hh_df.columns]
if missing_final_cols:
    print(f"警告：以下最终列在处理后的数据中缺失，将从分析中排除：{missing_final_cols}")

final_df = hh_df[existing_final_cols].copy()

# --- *** DIAGNOSE MISSING VALUES *** ---
print("\n--- 缺失值诊断 (回归变量) ---")
# Define the full set of variables intended for regression
regression_vars_to_check = ['debt_ratio_winsorized', 'log_debt_ratio_winsorized', 'head_siblings'] + head_controls + hh_controls
# Check only those that actually exist in final_df
regression_vars_existing = [col for col in regression_vars_to_check if col in final_df.columns]

missing_counts = final_df[regression_vars_existing].isnull().sum()
missing_percent = (missing_counts / len(final_df)) * 100
missing_stats = pd.DataFrame({'缺失数量': missing_counts, '缺失百分比 (%)': missing_percent})
missing_stats = missing_stats[missing_stats['缺失数量'] > 0].sort_values(by='缺失数量', ascending=False)

if not missing_stats.empty:
    print("回归分析前各变量的缺失值统计:")
    print(missing_stats)
    print(f"\n注意：总样本量为 {len(final_df)}。下面的回归将使用列表删除法处理这些缺失值。")
    # --- Optional: Code to drop variables with high missingness (Example) ---
    # threshold = 0.50 # Example: Drop if more than 50% missing
    # cols_to_drop = missing_stats[missing_stats['缺失百分比 (%)'] > threshold * 100].index.tolist()
    # # Ensure the core IV is not dropped accidentally
    # if 'head_siblings' in cols_to_drop:
    #     cols_to_drop.remove('head_siblings')
    # if cols_to_drop:
    #     print(f"\n基于 >{threshold*100}% 的缺失阈值，考虑移除以下变量: {cols_to_drop}")
    #     # final_df = final_df.drop(columns=cols_to_drop)
    #     # print("变量已移除。")
    # else:
    #     print("\n没有变量超过缺失阈值。")

else:
    print("回归变量中没有发现缺失值。")


# --- Final Cleaning for Regression (Listwise Deletion) ---
print("\n正在为回归分析执行最终清理 (列表删除法)...")
reg_independent_vars = ['head_siblings'] + head_controls + hh_controls
reg_independent_vars = [v for v in reg_independent_vars if v in final_df.columns] # Keep only existing columns

# Model 1 (Original DV)
key_reg_vars_model1 = ['debt_ratio_winsorized'] + reg_independent_vars
final_df_model1 = final_df.dropna(subset=[col for col in key_reg_vars_model1 if col in final_df.columns]).copy()
rows_after_model1 = len(final_df_model1)
print(f"模型1 (DV=debt_ratio_winsorized): 清理后用于回归的数据集包含 {rows_after_model1} 个家庭 (原始 {len(final_df)}).")
if len(final_df) > 0:
     print(f"数据丢失比例: {(len(final_df) - rows_after_model1) / len(final_df):.1%}")

# Model 2 (Log DV)
key_reg_vars_model2 = ['log_debt_ratio_winsorized'] + reg_independent_vars
final_df_model2 = final_df.dropna(subset=[col for col in key_reg_vars_model2 if col in final_df.columns]).copy()
if 'log_debt_ratio_winsorized' in final_df_model2.columns:
    final_df_model2 = final_df_model2[np.isfinite(final_df_model2['log_debt_ratio_winsorized'])]
rows_after_model2 = len(final_df_model2)
print(f"模型2 (DV=log_debt_ratio_winsorized): 清理后用于回归的数据集包含 {rows_after_model2} 个家庭。")


# --- Function to Calculate VIF ---
def calculate_vif(df, cols):
    x_vif = df[cols]
    x_vif = sm.add_constant(x_vif, has_constant='add') # Add constant for VIF calculation
    vif_data = pd.DataFrame()
    vif_data["feature"] = x_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(x_vif.values, i) for i in range(x_vif.shape[1])]
    # Exclude the constant's VIF for interpretation
    return vif_data[vif_data["feature"] != 'const'].sort_values('VIF', ascending=False)


# --- Regression Analysis ---

# --- Model 1: OLS (Baseline) ---
print("\n--- 回归分析: 模型 1 (OLS, DV = debt_ratio_winsorized) ---")
if rows_after_model1 < len(reg_independent_vars) + 2:
     print(f"错误：模型1 清理后剩余数据过少 ({rows_after_model1} obs)，无法执行回归分析。")
else:
    Y1 = final_df_model1['debt_ratio_winsorized']
    X1_cols_for_reg = [col for col in reg_independent_vars if col in final_df_model1.columns]
    X1 = final_df_model1[X1_cols_for_reg]
    X1_const = sm.add_constant(X1)

    try:
        model1 = sm.OLS(Y1, X1_const)
        results1 = model1.fit()
        print("\n模型 1: OLS 结果")
        print(results1.summary())

        # VIF Calculation
        print("\n模型 1: VIF 诊断")
        vif_results1 = calculate_vif(final_df_model1, X1_cols_for_reg)
        print(vif_results1)
        high_vif = vif_results1[vif_results1['VIF'] > 5] # Common threshold
        if not high_vif.empty:
            print(f"警告: 模型 1 中以下变量 VIF > 5，表明可能存在多重共线性: {high_vif['feature'].tolist()}")
        else:
            print("模型 1 中未发现 VIF > 5 的变量。")

    except Exception as e:
        print(f"模型 1 OLS 回归执行失败: {e}")

# --- Model 2: OLS (Log DV, Baseline) ---
print("\n--- 回归分析: 模型 2 (OLS, DV = log_debt_ratio_winsorized) ---")
if rows_after_model2 < len(reg_independent_vars) + 2:
     print(f"错误：模型2 清理后剩余数据过少 ({rows_after_model2} obs)，无法执行回归分析。")
elif 'log_debt_ratio_winsorized' not in final_df_model2.columns:
     print("错误：模型2 的因变量 'log_debt_ratio_winsorized' 不存在。")
else:
    Y2 = final_df_model2['log_debt_ratio_winsorized']
    X2_cols_for_reg = [col for col in reg_independent_vars if col in final_df_model2.columns]
    X2 = final_df_model2[X2_cols_for_reg]
    X2_const = sm.add_constant(X2)

    try:
        model2 = sm.OLS(Y2, X2_const)
        results2 = model2.fit()
        print("\n模型 2: OLS 结果")
        print(results2.summary())

        # VIF Calculation
        print("\n模型 2: VIF 诊断")
        vif_results2 = calculate_vif(final_df_model2, X2_cols_for_reg)
        print(vif_results2)
        high_vif2 = vif_results2[vif_results2['VIF'] > 5]
        if not high_vif2.empty:
            print(f"警告: 模型 2 中以下变量 VIF > 5，表明可能存在多重共线性: {high_vif2['feature'].tolist()}")
        else:
            print("模型 2 中未发现 VIF > 5 的变量。")

    except Exception as e:
        print(f"模型 2 OLS 回归执行失败: {e}")


# --- Model 3: Ridge Regression (Handles Multicollinearity) ---
print("\n--- 回归分析: 模型 3 (RidgeCV, DV = debt_ratio_winsorized) ---")
if rows_after_model1 < len(reg_independent_vars) + 2:
     print(f"错误：模型3 (Ridge) 数据过少 ({rows_after_model1} obs)，无法执行。")
else:
    Y1_ridge = final_df_model1['debt_ratio_winsorized']
    X1_ridge_cols = [col for col in reg_independent_vars if col in final_df_model1.columns]
    X1_ridge = final_df_model1[X1_ridge_cols]

    # Scale features for Ridge
    scaler = StandardScaler()
    X1_ridge_scaled = scaler.fit_transform(X1_ridge)

    # Define alphas for RidgeCV (logarithmic scale is common)
    alphas = np.logspace(-6, 6, 13) # Example range

    try:
        # Use RidgeCV to find the best alpha
        ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
        ridge_cv.fit(X1_ridge_scaled, Y1_ridge)

        print(f"\n模型 3: RidgeCV 结果")
        print(f"最佳 Alpha: {ridge_cv.alpha_}")
        # Store coefficients with names
        ridge_coefs = pd.Series(ridge_cv.coef_, index=X1_ridge_cols)
        print("岭回归系数 (基于标准化数据):")
        print(ridge_coefs.sort_values(ascending=False))
        # Note: Intercept is calculated separately in RidgeCV
        print(f"截距 (Intercept): {ridge_cv.intercept_}")
        # R-squared can be calculated using score method
        print(f"岭回归 R-squared: {ridge_cv.score(X1_ridge_scaled, Y1_ridge):.4f}")

    except Exception as e:
        print(f"模型 3 Ridge 回归执行失败: {e}")

# --- Model 4: Ridge Regression (Log DV) ---
print("\n--- 回归分析: 模型 4 (RidgeCV, DV = log_debt_ratio_winsorized) ---")
if rows_after_model2 < len(reg_independent_vars) + 2:
     print(f"错误：模型4 (Ridge LogDV) 数据过少 ({rows_after_model2} obs)，无法执行。")
elif 'log_debt_ratio_winsorized' not in final_df_model2.columns:
     print("错误：模型4 的因变量 'log_debt_ratio_winsorized' 不存在。")
else:
    Y2_ridge = final_df_model2['log_debt_ratio_winsorized']
    X2_ridge_cols = [col for col in reg_independent_vars if col in final_df_model2.columns]
    X2_ridge = final_df_model2[X2_ridge_cols]

    # Scale features (can reuse scaler if fitted on same columns, but safer to refit)
    scaler2 = StandardScaler()
    X2_ridge_scaled = scaler2.fit_transform(X2_ridge)

    alphas = np.logspace(-6, 6, 13)

    try:
        ridge_cv2 = RidgeCV(alphas=alphas, store_cv_values=True)
        ridge_cv2.fit(X2_ridge_scaled, Y2_ridge)

        print(f"\n模型 4: RidgeCV 结果 (Log DV)")
        print(f"最佳 Alpha: {ridge_cv2.alpha_}")
        ridge_coefs2 = pd.Series(ridge_cv2.coef_, index=X2_ridge_cols)
        print("岭回归系数 (基于标准化数据):")
        print(ridge_coefs2.sort_values(ascending=False))
        print(f"截距 (Intercept): {ridge_cv2.intercept_}")
        print(f"岭回归 R-squared: {ridge_cv2.score(X2_ridge_scaled, Y2_ridge):.4f}")

    except Exception as e:
        print(f"模型 4 Ridge 回归执行失败: {e}")


# --- Model 5: Robust Regression (Handles Outliers/Non-Normal Errors) ---
print("\n--- 回归分析: 模型 5 (RLM, DV = debt_ratio_winsorized) ---")
if rows_after_model1 < len(reg_independent_vars) + 2:
     print(f"错误：模型5 (RLM) 数据过少 ({rows_after_model1} obs)，无法执行。")
else:
    Y1_rlm = final_df_model1['debt_ratio_winsorized']
    X1_rlm_cols = [col for col in reg_independent_vars if col in final_df_model1.columns]
    X1_rlm = final_df_model1[X1_rlm_cols]
    X1_rlm_const = sm.add_constant(X1_rlm)

    try:
        # Use Huber's T norm for robustness
        rlm_model = sm.RLM(Y1_rlm, X1_rlm_const, M=sm.robust.norms.HuberT())
        rlm_results = rlm_model.fit()
        print("\n模型 5: RLM (HuberT) 结果")
        print(rlm_results.summary())
        # Note: RLM summary doesn't provide R-squared directly in the same way OLS does.
        # Pseudo R-squared can be calculated if needed, but interpretation differs.

    except Exception as e:
        print(f"模型 5 RLM 回归执行失败: {e}")


# --- Display Summary of Final Datasets Used in Regression ---
print("\n--- 清理后用于回归的数据集描述性统计 ---")
print("\n模型1, 3, 5 使用的数据集:")
if rows_after_model1 > 0:
    with pd.option_context('display.float_format', '{:,.4f}'.format):
        print(final_df_model1[[col for col in key_reg_vars_model1 if col in final_df_model1.columns]].describe())
else:
    print("数据集为空。")

print("\n模型2, 4 使用的数据集:")
if rows_after_model2 > 0:
    with pd.option_context('display.float_format', '{:,.4f}'.format):
        print(final_df_model2[[col for col in key_reg_vars_model2 if col in final_df_model2.columns]].describe())
else:
    print("数据集为空。")


# --- Optional: Save Processed Data ---
if output_file_path:
    try:
        # Save the dataframe *before* dropping NAs for regression
        # final_df contains the new log columns and midpoint calculations
        final_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"\n包含所有计算和诊断列的处理后数据已保存至 {output_file_path}")
    except Exception as e:
        print(f"\n保存处理后的数据时出错: {e}")

print("\n--- 数据处理与分析完毕 ---")