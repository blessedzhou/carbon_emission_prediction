import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import joblib
# from xgboost import XGBClassifier

# from xgboost import XGBModel


import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")
import matplotlib

matplotlib.use('Agg')
import seaborn as sns

# from sklearn.preprocessing import StandardScaler

# warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

from streamlit_option_menu import option_menu

selected_menu = option_menu(
    menu_title=None,
    options=["Home", "About The Developers", "Contact"],
    icons=["house", "person-workspace", "envelope"],
    menu_icon="cast",
    orientation="horizontal"
)


# Load data
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df


# Get the keys from the dictionary
def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value


# Find the key from the dictionary
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


data = load_data("data/german_credit_data_3.csv")

model = joblib.load('files/My_model3.joblib')
# model = XGBClassifier(learning_rate = 0.1,
#                              max_depth= 3,
#                              min_samples_split= 0.1,
#                             n_estimators = 100,
#                              subsample= 0.9).fit(X,y)


Sex_label = {'male': 1, 'female': 0}

Purpose_label = {'radio/TV': 5,
                 'education': 3,
                 'furniture/equipment': 4,
                 'car': 1, 'business': 0,
                 'domestic appliances': 2,
                 'repairs': 6,
                 'vacation/others': 7}

Housing_label = {'own': 1, 'free': 0, 'rent': 2}

Saving_accounts_label = {'little': 0, 'quite rich': 2, 'rich': 3, 'moderate': 1}

Checking_account_label = {'little': 0, 'moderate': 1, 'rich': 2}

class_label = {'Good': 0, 'Bad': 1}

# -------------------------------------PAGE 1 ----------------------------------------------------
nav = st.sidebar.radio("Navigation", ["Client", "Admin"])
if selected_menu == "Home":

    # -------------------------------------Client page ----------------------------------------------------
    if nav == "Client":
        st.subheader("Carbon emission prediction")

        # defining variables
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            SulphurDioxide_SO2_column_number_density = st.number_input("SulphurDioxide_SO2_column_number_density")
            SulphurDioxide_SO2_column_number_density_amf = st.number_input(
                "SulphurDioxide_SO2_column_number_density_amf")
            SulphurDioxide_SO2_slant_column_number_density = st.number_input(
                "SulphurDioxide_SO2_slant_column_number_density")
            SulphurDioxide_cloud_fraction = st.number_input("SulphurDioxide_cloud_fraction")
            SulphurDioxide_sensor_azimuth_angle = st.number_input("SulphurDioxide_sensor_azimuth_angle")
            SulphurDioxide_sensor_zenith_angle = st.number_input("SulphurDioxide_sensor_zenith_angle")
            SulphurDioxide_solar_azimuth_angle = st.number_input("SulphurDioxide_solar_azimuth_angle")
            SulphurDioxide_solar_zenith_angle = st.number_input("SulphurDioxide_solar_zenith_angle")
            SulphurDioxide_SO2_column_number_density_15km = st.number_input(
                "SulphurDioxide_SO2_column_number_density_15km")
            CarbonMonoxide_CO_column_number_density = st.number_input("CarbonMonoxide_CO_column_number_density")
            CarbonMonoxide_H2O_column_number_density = st.number_input("CarbonMonoxide_H2O_column_number_density")
            CarbonMonoxide_cloud_height = st.number_input("CarbonMonoxide_cloud_height")
            CarbonMonoxide_sensor_altitude = st.number_input("CarbonMonoxide_sensor_altitude")
            CarbonMonoxide_sensor_azimuth_angle = st.number_input("CarbonMonoxide_sensor_azimuth_angle")
            CarbonMonoxide_sensor_zenith_angle = st.number_input("CarbonMonoxide_sensor_zenith_angle")
            CarbonMonoxide_solar_azimuth_angle = st.number_input("CarbonMonoxide_solar_azimuth_angle")
            CarbonMonoxide_solar_zenith_angle = st.number_input("CarbonMonoxide_solar_zenith_angle")
            NitrogenDioxide_NO2_column_number_density = st.number_input("NitrogenDioxide_NO2_column_number_density")

        with col2:
            NitrogenDioxide_tropospheric_NO2_column_number_density = st.number_input(
                "NitrogenDioxide_tropospheric_NO2_column_number_density")
            NitrogenDioxide_stratospheric_NO2_column_number_density = st.number_input(
                "NitrogenDioxide_stratospheric_NO2_column_number_density")
            NitrogenDioxide_NO2_slant_column_number_density = st.number_input(
                "NitrogenDioxide_NO2_slant_column_number_density")
            NitrogenDioxide_tropopause_pressure = st.number_input("NitrogenDioxide_tropopause_pressure")
            NitrogenDioxide_absorbing_aerosol_index = st.number_input("NitrogenDioxide_absorbing_aerosol_index")
            NitrogenDioxide_cloud_fraction = st.number_input("NitrogenDioxide_cloud_fraction")
            NitrogenDioxide_sensor_altitude = st.number_input("NitrogenDioxide_sensor_altitude")
            NitrogenDioxide_sensor_azimuth_angle = st.number_input("NitrogenDioxide_sensor_azimuth_angle")
            NitrogenDioxide_sensor_zenith_angle = st.number_input("NitrogenDioxide_sensor_zenith_angle")
            NitrogenDioxide_solar_azimuth_angle = st.number_input("NitrogenDioxide_solar_azimuth_angle")
            NitrogenDioxide_solar_zenith_angle = st.number_input("NitrogenDioxide_solar_zenith_angle")
            Formaldehyde_tropospheric_HCHO_column_number_density = st.number_input(
                "Formaldehyde_tropospheric_HCHO_column_number_density")
            Formaldehyde_tropospheric_HCHO_column_number_density_amf = st.number_input(
                "Formaldehyde_tropospheric_HCHO_column_number_density_amf")
            Formaldehyde_HCHO_slant_column_number_density = st.number_input(
                "Formaldehyde_HCHO_slant_column_number_density")
            Formaldehyde_cloud_fraction = st.number_input(
                "Formaldehyde_cloud_fraction")
            Formaldehyde_solar_zenith_angle = st.number_input("Formaldehyde_solar_zenith_angle")
            Formaldehyde_solar_azimuth_angle = st.number_input("Formaldehyde_solar_azimuth_angle")
            Formaldehyde_sensor_zenith_angle = st.number_input("Formaldehyde_sensor_zenith_angle")

        with col3:
            Formaldehyde_sensor_azimuth_angle = st.number_input("Formaldehyde_sensor_azimuth_angle")
            UvAerosolIndex_absorbing_aerosol_index = st.number_input("UvAerosolIndex_absorbing_aerosol_index")
            UvAerosolIndex_sensor_altitude = st.number_input("UvAerosolIndex_sensor_altitude")
            UvAerosolIndex_sensor_azimuth_angle = st.number_input("UvAerosolIndex_sensor_azimuth_angle")
            UvAerosolIndex_sensor_zenith_angle = st.number_input("UvAerosolIndex_sensor_zenith_angle")
            UvAerosolIndex_solar_azimuth_angle = st.number_input("UvAerosolIndex_solar_azimuth_angle")
            UvAerosolIndex_solar_zenith_angle = st.number_input("UvAerosolIndex_solar_zenith_angle")
            Ozone_O3_column_number_density = st.number_input("Ozone_O3_column_number_density")
            Ozone_O3_column_number_density_amf = st.number_input("Ozone_O3_column_number_density_amf")
            Ozone_O3_slant_column_number_density = st.number_input("Ozone_O3_slant_column_number_density")
            Ozone_O3_effective_temperature = st.number_input("Ozone_O3_effective_temperature")
            Ozone_cloud_fraction = st.number_input("Ozone_cloud_fraction")
            Ozone_sensor_azimuth_angle = st.number_input("Ozone_sensor_azimuth_angle")
            Ozone_sensor_zenith_angle = st.number_input("Ozone_sensor_zenith_angle")
            Ozone_solar_azimuth_angle = st.number_input("Ozone_solar_azimuth_angle")
            Ozone_solar_zenith_angle = st.number_input("Ozone_solar_zenith_angle")
            UvAerosolLayerHeight_aerosol_height = st.number_input("UvAerosolLayerHeight_aerosol_height")
            UvAerosolLayerHeight_aerosol_pressure = st.number_input("UvAerosolLayerHeight_aerosol_pressure")

        with col4:
            UvAerosolLayerHeight_aerosol_optical_depth = st.number_input("UvAerosolLayerHeight_aerosol_optical_depth")
            UvAerosolLayerHeight_sensor_zenith_angle = st.number_input("UvAerosolLayerHeight_sensor_zenith_angle")
            UvAerosolLayerHeight_sensor_azimuth_angle = st.number_input("UvAerosolLayerHeight_sensor_azimuth_angle")
            UvAerosolLayerHeight_solar_azimuth_angle = st.number_input("UvAerosolLayerHeight_solar_azimuth_angle")
            UvAerosolLayerHeight_solar_zenith_angle = st.number_input("UvAerosolLayerHeight_solar_zenith_angle")
            Cloud_cloud_fraction = st.number_input("Cloud_cloud_fraction")
            Cloud_cloud_top_pressure = st.number_input("Cloud_cloud_top_pressure")
            Cloud_cloud_top_height = st.number_input("Cloud_cloud_top_height")
            Cloud_cloud_base_pressure = st.number_input("Cloud_cloud_base_pressure")
            Cloud_cloud_base_height = st.number_input("Cloud_cloud_base_height")
            Cloud_cloud_optical_depth = st.number_input("Cloud_cloud_optical_depth")
            Cloud_surface_albedo = st.number_input("Cloud_surface_albedo")
            Cloud_sensor_azimuth_angle = st.number_input("Cloud_sensor_azimuth_angle")
            Cloud_sensor_zenith_angle = st.number_input("Cloud_sensor_zenith_angle")
            Cloud_solar_azimuth_angle = st.number_input("Cloud_solar_azimuth_angle")
            Cloud_solar_zenith_angle = st.number_input("Cloud_solar_zenith_angle")

            # Encoded values
            # c_sex = get_value(sex, Sex_label)
            # c_housing = get_value(housing, Housing_label)
            # c_saving_accounts = get_value(saving_accounts, Saving_accounts_label)
            # c_checking_account = get_value(checking_account, Checking_account_label)

            # pretty_data = {
            #     "Age": age,
            #     "Sex": sex,
            #     "Housing": housing,
            #     "Saving accounts": saving_accounts,
            #     "Checking accounts": checking_account,
            #     "Credit amount": credit_amount,
            #     "Duration": duration
            # }

        sample_data = [
            (SulphurDioxide_SO2_column_number_density),
            (SulphurDioxide_SO2_column_number_density_amf),
            (SulphurDioxide_SO2_slant_column_number_density),
            (SulphurDioxide_cloud_fraction),
            (SulphurDioxide_sensor_azimuth_angle),
            (SulphurDioxide_sensor_zenith_angle),
            (SulphurDioxide_solar_azimuth_angle),
            (SulphurDioxide_solar_zenith_angle),
            (SulphurDioxide_SO2_column_number_density_15km),
            (CarbonMonoxide_CO_column_number_density),
            (CarbonMonoxide_H2O_column_number_density),
            (CarbonMonoxide_cloud_height),
            (CarbonMonoxide_sensor_altitude),
            (CarbonMonoxide_sensor_azimuth_angle),
            (CarbonMonoxide_sensor_zenith_angle),
            (CarbonMonoxide_solar_azimuth_angle),
            (CarbonMonoxide_solar_zenith_angle),
            (NitrogenDioxide_NO2_column_number_density),
            (NitrogenDioxide_tropospheric_NO2_column_number_density),
            (NitrogenDioxide_stratospheric_NO2_column_number_density),
            (NitrogenDioxide_NO2_slant_column_number_density),
            (NitrogenDioxide_tropopause_pressure),
            (NitrogenDioxide_absorbing_aerosol_index),
            (NitrogenDioxide_cloud_fraction),
            (NitrogenDioxide_sensor_altitude),
            (NitrogenDioxide_sensor_azimuth_angle),
            (NitrogenDioxide_sensor_zenith_angle),
            (NitrogenDioxide_solar_azimuth_angle),
            (NitrogenDioxide_solar_zenith_angle),
            (Formaldehyde_tropospheric_HCHO_column_number_density),
            (Formaldehyde_tropospheric_HCHO_column_number_density_amf),
            (Formaldehyde_HCHO_slant_column_number_density),
            (Formaldehyde_cloud_fraction),
            (Formaldehyde_solar_zenith_angle),
            (Formaldehyde_solar_azimuth_angle),
            (Formaldehyde_sensor_zenith_angle),
            (Formaldehyde_sensor_azimuth_angle),
            (UvAerosolIndex_absorbing_aerosol_index),
            (UvAerosolIndex_sensor_altitude),
            (UvAerosolIndex_sensor_azimuth_angle),
            (UvAerosolIndex_sensor_zenith_angle),
            (UvAerosolIndex_solar_azimuth_angle),
            (UvAerosolIndex_solar_zenith_angle),
            (Ozone_O3_column_number_density),
            (Ozone_O3_column_number_density_amf),
            (Ozone_O3_slant_column_number_density),
            (Ozone_O3_effective_temperature),
            (Ozone_cloud_fraction),
            (Ozone_sensor_azimuth_angle),
            (Ozone_sensor_zenith_angle),
            (Ozone_solar_azimuth_angle),
            (Ozone_solar_zenith_angle),
            (UvAerosolLayerHeight_aerosol_height),
            (UvAerosolLayerHeight_aerosol_pressure),
            (UvAerosolLayerHeight_aerosol_optical_depth),
            (UvAerosolLayerHeight_sensor_zenith_angle),
            (UvAerosolLayerHeight_sensor_azimuth_angle),
            (UvAerosolLayerHeight_solar_azimuth_angle),
            (UvAerosolLayerHeight_solar_zenith_angle),
            (Cloud_cloud_fraction),
            (Cloud_cloud_top_pressure),
            (Cloud_cloud_top_height),
            (Cloud_cloud_base_pressure),
            (Cloud_cloud_base_height),
            (Cloud_cloud_optical_depth),
            (Cloud_surface_albedo),
            (Cloud_sensor_azimuth_angle),
            (Cloud_sensor_zenith_angle),
            (Cloud_solar_azimuth_angle),
            (Cloud_solar_zenith_angle)]


        shaped_data = np.array(sample_data).reshape(1, -1)

        if st.button("Predict"):
            predictor = model
            prediction = predictor.predict(shaped_data)

            result = prediction
            predicted_result = result

            if predicted_result < 100:
                st.success(predicted_result)

            else:
                st.error(predicted_result[0])
                st.error(
                    "Carbon emission is higher than the internationally approved level. Let's reduce global warming together")

        if st.button("Save_information"):
            predictor = model
            prediction = predictor.predict(shaped_data)

            result = prediction[0]
            predicted_result = get_key(result, class_label)

            # to_add = {
            #     "Age": [age],
            #     "Sex": [sex],
            #     "Housing": [housing],
            #     "Saving accounts": [saving_accounts],
            #     "Checking accounts": [checking_account],
            #     "Credit amount": [checking_account],
            #     "Duration": [duration],
            #     "Decision": [predicted_result]
            # }
            to_add = {
                'SulphurDioxide_SO2_column_number_density': [SulphurDioxide_SO2_column_number_density],
                 'SulphurDioxide_SO2_column_number_density_amf': [SulphurDioxide_SO2_column_number_density_amf],
                 'SulphurDioxide_SO2_slant_column_number_density': [SulphurDioxide_SO2_slant_column_number_density],
                 'SulphurDioxide_cloud_fraction': [SulphurDioxide_cloud_fraction],
                 'SulphurDioxide_sensor_azimuth_angle': [SulphurDioxide_sensor_azimuth_angle],
                 'SulphurDioxide_sensor_zenith_angle': [SulphurDioxide_sensor_zenith_angle],
                 'SulphurDioxide_solar_azimuth_angle': [SulphurDioxide_solar_azimuth_angle],
                 'SulphurDioxide_solar_zenith_angle': [SulphurDioxide_solar_zenith_angle],
                 'SulphurDioxide_SO2_column_number_density_15km': [SulphurDioxide_SO2_column_number_density_15km],
                 'CarbonMonoxide_CO_column_number_density': [CarbonMonoxide_CO_column_number_density],
                 'CarbonMonoxide_H2O_column_number_density': [CarbonMonoxide_H2O_column_number_density],
                 'CarbonMonoxide_cloud_height': [CarbonMonoxide_cloud_height],
                 'CarbonMonoxide_sensor_altitude': [CarbonMonoxide_sensor_altitude],
                 'CarbonMonoxide_sensor_azimuth_angle': [CarbonMonoxide_sensor_azimuth_angle],
                 'CarbonMonoxide_sensor_zenith_angle': [CarbonMonoxide_sensor_zenith_angle],
                 'CarbonMonoxide_solar_azimuth_angle': [CarbonMonoxide_solar_azimuth_angle],
                 'CarbonMonoxide_solar_zenith_angle': [CarbonMonoxide_solar_zenith_angle],
                 'NitrogenDioxide_NO2_column_number_density': [NitrogenDioxide_NO2_column_number_density],
                 'NitrogenDioxide_tropospheric_NO2_column_number_density': [NitrogenDioxide_tropospheric_NO2_column_number_density],
                 'NitrogenDioxide_stratospheric_NO2_column_number_density': [NitrogenDioxide_stratospheric_NO2_column_number_density],
                 'NitrogenDioxide_NO2_slant_column_number_density': [NitrogenDioxide_NO2_slant_column_number_density],
                 'NitrogenDioxide_tropopause_pressure': [NitrogenDioxide_tropopause_pressure],
                 'NitrogenDioxide_absorbing_aerosol_index': [NitrogenDioxide_absorbing_aerosol_index],
                 'NitrogenDioxide_cloud_fraction': [NitrogenDioxide_cloud_fraction],
                 'NitrogenDioxide_sensor_altitude': [NitrogenDioxide_sensor_altitude],
                 'NitrogenDioxide_sensor_azimuth_angle': [NitrogenDioxide_sensor_azimuth_angle],
                 'NitrogenDioxide_sensor_zenith_angle': [NitrogenDioxide_sensor_zenith_angle],
                 'NitrogenDioxide_solar_azimuth_angle': [NitrogenDioxide_solar_azimuth_angle],
                 'NitrogenDioxide_solar_zenith_angle': [NitrogenDioxide_solar_zenith_angle],
                 'Formaldehyde_tropospheric_HCHO_column_number_density': [Formaldehyde_tropospheric_HCHO_column_number_density],
                 'Formaldehyde_tropospheric_HCHO_column_number_density_amf': [Formaldehyde_tropospheric_HCHO_column_number_density_amf],
                 'Formaldehyde_HCHO_slant_column_number_density': [Formaldehyde_HCHO_slant_column_number_density],
                 'Formaldehyde_cloud_fraction': [Formaldehyde_cloud_fraction],
                 'Formaldehyde_solar_zenith_angle': [Formaldehyde_solar_zenith_angle],
                 'Formaldehyde_solar_azimuth_angle': [Formaldehyde_solar_azimuth_angle],
                 'Formaldehyde_sensor_zenith_angle': [Formaldehyde_sensor_zenith_angle],
                 'Formaldehyde_sensor_azimuth_angle': [Formaldehyde_sensor_azimuth_angle],
                 'UvAerosolIndex_absorbing_aerosol_index': [UvAerosolIndex_absorbing_aerosol_index],
                 'UvAerosolIndex_sensor_altitude': [UvAerosolIndex_sensor_altitude],
                 'UvAerosolIndex_sensor_azimuth_angle': [UvAerosolIndex_sensor_azimuth_angle],
                 'UvAerosolIndex_sensor_zenith_angle': [UvAerosolIndex_sensor_zenith_angle],
                 'UvAerosolIndex_solar_azimuth_angle': [UvAerosolIndex_solar_azimuth_angle],
                 'UvAerosolIndex_solar_zenith_angle': [UvAerosolIndex_solar_zenith_angle],
                 'Ozone_O3_column_number_density': [Ozone_O3_column_number_density],
                 'Ozone_O3_column_number_density_amf': [Ozone_O3_column_number_density_amf],
                 'Ozone_O3_slant_column_number_density': [Ozone_O3_slant_column_number_density],
                 'Ozone_O3_effective_temperature': [Ozone_O3_effective_temperature],
                 'Ozone_cloud_fraction': [Ozone_cloud_fraction],
                 'Ozone_sensor_azimuth_angle': [Ozone_sensor_azimuth_angle],
                 'Ozone_sensor_zenith_angle': [Ozone_sensor_zenith_angle],
                 'Ozone_solar_azimuth_angle': [Ozone_solar_azimuth_angle],
                 'Ozone_solar_zenith_angle': [Ozone_solar_zenith_angle],
                 'UvAerosolLayerHeight_aerosol_height': [UvAerosolLayerHeight_aerosol_height],
                 'UvAerosolLayerHeight_aerosol_pressure': [UvAerosolLayerHeight_aerosol_pressure],
                 'UvAerosolLayerHeight_aerosol_optical_depth': [UvAerosolLayerHeight_aerosol_optical_depth],
                 'UvAerosolLayerHeight_sensor_zenith_angle': [UvAerosolLayerHeight_sensor_zenith_angle],
                 'UvAerosolLayerHeight_sensor_azimuth_angle': [UvAerosolLayerHeight_sensor_azimuth_angle],
                 'UvAerosolLayerHeight_solar_azimuth_angle': [UvAerosolLayerHeight_solar_azimuth_angle],
                 'UvAerosolLayerHeight_solar_zenith_angle': [UvAerosolLayerHeight_solar_zenith_angle],
                 'Cloud_cloud_fraction': [Cloud_cloud_fraction],
                 'Cloud_cloud_top_pressure': [Cloud_cloud_top_pressure],
                 'Cloud_cloud_top_height': [Cloud_cloud_top_height],
                 'Cloud_cloud_base_pressure': [Cloud_cloud_base_pressure],
                 'Cloud_cloud_base_height': [Cloud_cloud_base_height],
                 'Cloud_cloud_optical_depth': [Cloud_cloud_optical_depth],
                 'Cloud_surface_albedo': [Cloud_surface_albedo],
                 'Cloud_sensor_azimuth_angle': [Cloud_sensor_azimuth_angle],
                 'Cloud_sensor_zenith_angle': [Cloud_sensor_zenith_angle],
                 'Cloud_solar_azimuth_angle': [Cloud_solar_azimuth_angle],
                 'Cloud_solar_zenith_angle': [Cloud_solar_zenith_angle],
                'emission_level': [result]
                }

            to_add = pd.DataFrame(to_add)
            to_add.to_csv("data/New_entry_Data.csv", mode='a', header=False, index=False)
            st.success("Saved")

# -------------------------------------Admin page ----------------------------------------------------
if nav == "Admin":

    # dfx_og = pd.read_csv(r"data/data.csv", index_col=0)
    #
    # # for EDA
    # column_obj = [dt for dt in data.columns if data[dt].dtype == "O"]
    #
    # dfx = dfx_og.sort_values(by="prob_Good", ascending=False)
    #
    # # turning deciles
    # dfx["Deciles"] = pd.qcut(dfx["prob_Bad"], 10, labels=np.arange(1, 11, 1))
    # dfx["Count"] = 1
    #
    # dfx = dfx.sort_values(by="Deciles", ascending=False)
    #
    # dfx["prob_Bad"] = dfx["prob_Bad"] * 100
    # dfx["prob_Good"] = dfx["prob_Good"] * 100
    #
    # # rounding the percentages
    # dfx = dfx.round({"prob_Bad": 2, "prob_Good": 2})
    #
    # pivot_table = pd.pivot_table(dfx, index="Deciles", values=["predicted", "prob_Good", "Count"],
    #                              aggfunc={"predicted": sum,
    #                                       "prob_Good": min,
    #                                       "Count": pd.Series.count})
    #
    # pivot_table.rename(columns={"predicted": "Bad", "Count": "Total"}, inplace=True)
    #
    # pivot_table["Good"] = pivot_table["Total"] - pivot_table["Bad"]
    # pivot_table["Cumm_Good"] = pivot_table["Good"].cumsum()
    # pivot_table["Cumm_Bad"] = pivot_table["Bad"].cumsum()
    # pivot_table["Cumm_Bad %"] = 100 * (pivot_table["Bad"].cumsum() / pivot_table["Bad"].sum())
    # pivot_table["Cumm_Good %"] = 100 * (pivot_table["Good"].cumsum() / pivot_table["Good"].sum())
    # pivot_table["Cumm_Bad_Avoided %"] = 100 - pivot_table["Cumm_Bad %"]
    #
    # if st.checkbox("Data Destribution"):
    #     dd_choice = st.selectbox("Destribute by",
    #                              ["None", "Sex", "Housing", "Saving accounts", "Checking account", "Risk",
    #                               "Duration"])
    #
    #     if dd_choice == "None":
    #         st.write(" ")
    #
    #     if dd_choice == "Sex":
    #         plt.figure(figsize=(6, 3))
    #         plt.bar(data["Sex"].value_counts().index, data["Sex"].value_counts())
    #         st.pyplot()
    #
    #     elif dd_choice == "Housing":
    #         plt.figure(figsize=(6, 3))
    #         plt.bar(data["Housing"].value_counts().index, data["Housing"].value_counts())
    #         st.pyplot()
    #
    #     elif dd_choice == "Saving accounts":
    #         plt.figure(figsize=(6, 3))
    #         plt.bar(data["Saving accounts"].value_counts().index, data["Saving accounts"].value_counts())
    #         st.pyplot()
    #
    #     elif dd_choice == "Checking account	":
    #         plt.figure(figsize=(6, 3))
    #         plt.bar(data["Checking account"].value_counts().index, data["Checking account"].value_counts())
    #         st.pyplot()
    #
    #     elif dd_choice == "Risk":
    #         plt.figure(figsize=(6, 3))
    #         plt.bar(data["Risk"].value_counts().index, data["Risk"].value_counts())
    #         st.pyplot()
    #
    #     elif dd_choice == "Duration":
    #         plt.figure(figsize=(9, 3))
    #         plt.plot(data["Duration"], linewidth=1)
    #         st.pyplot()
    #
    # if st.checkbox("Description of Distribuition Risk by"):
    #     rd_choice = st.selectbox("Destribute by", ["None", "Sex", "Housing", "Saving accounts", "Checking account"])
    #
    #     if rd_choice == "None":
    #         st.write(" ")
    #
    #     if rd_choice == "Sex":
    #         plt.figure(figsize=(6, 3))
    #         g = sns.countplot(x="Sex", data=data, palette="husl", hue="Risk")
    #         g.set_title("Sex Count", fontsize=15)
    #         g.set_xlabel("Sex type", fontsize=12)
    #         g.set_ylabel("Count", fontsize=12)
    #         st.pyplot()
    #
    #     if rd_choice == "Saving accounts":
    #         plt.figure(figsize=(6, 3))
    #         g = sns.countplot(x="Saving accounts", data=data, palette="husl", hue="Risk")
    #         g.set_title("Saving Accounts Count", fontsize=15)
    #         g.set_xlabel("Saving Accounts type", fontsize=12)
    #         g.set_ylabel("Count", fontsize=12)
    #         st.pyplot()
    #
    #     if rd_choice == "Housing":
    #         plt.figure(figsize=(6, 3))
    #         g = sns.countplot(x="Housing", data=data, palette="husl", hue="Risk")
    #         g.set_title("Housing Count", fontsize=15)
    #         g.set_xlabel("Housing type", fontsize=12)
    #         g.set_ylabel("Count", fontsize=12)
    #         st.pyplot()
    #
    #     if rd_choice == "Checking account":
    #         plt.figure(figsize=(6, 3))
    #         g = sns.countplot(x="Checking account", data=data, palette="husl", hue="Risk")
    #         g.set_title("Checking account Count", fontsize=15)
    #         g.set_xlabel("Checking account type", fontsize=12)
    #         g.set_ylabel("Count", fontsize=12)
    #         st.pyplot()
    #
    # if st.checkbox("Show Current Credit Status"):
    #     st.dataframe(pivot_table)

    if st.checkbox("Show Added data"):
        new_entries = pd.read_csv("data/New_entry_Data.csv")
        st.dataframe(new_entries)


# -------------------------------------PAGE 2 ----------------------------------------------------

elif selected_menu == "About The Developers":
    st.write("BLESSED ZHOU R205757M")
    st.write("TONDERAI B MUTOMBWA R204739S")
    st.write("CHIKOMBORERO MUSHATI R207229H")
    st.write("CLADIOUS MUTAVIRWA R207523C")
    st.write("TINOMUKUDZA MABEZA R205535Q ")

# -------------------------------------PAGE 3 ----------------------------------------------------
elif selected_menu == "Contact":
    st.write(
        "Calls     : +263 776 464 136/ +263 786 591 814         \nWhatsApp : +263 776 464 136/ +263 786 591 814        \nEmail : brandonmutombwa@gmail.com / zhoublessed5@gmail.com")
