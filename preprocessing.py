import numpy as np
import pandas as pd
import logging


def get_preprocessing_employee_data(general_data, employee_survey, manager_survey):
    """
    Join and preprocess employee data from general, employee survey, and manager survey datasets.
    """
    try:
        logging.info("Processing employee data...")
        # Join all employee data
        employee_data_joined = general_data.join(
            employee_survey.set_index("EmployeeID"), on="EmployeeID", how="left"
        )
        employee_data_joined = employee_data_joined.join(
            manager_survey.set_index("EmployeeID"), on="EmployeeID", how="left"
        )
        employee_data_joined = employee_data_joined.set_index("EmployeeID")
        # Encode the 'Attrition' column
        employee_data_joined["Attrition"] = employee_data_joined["Attrition"].apply(
            lambda x: 1 if x == "Yes" else 0
        )
        employee_data_joined["MonthlyIncome_Squared"] = (
            employee_data_joined["MonthlyIncome"] ** 2
        )
        employee_data_joined["DistanceFromHome_Squared"] = (
            employee_data_joined["DistanceFromHome"] ** 2
        )
        # Drop unnecessary columns
        employee_data_joined.drop(
            columns=[
                "Over18",
                "EmployeeCount",
                "StandardHours",
                "MonthlyIncome",
                "DistanceFromHome",
            ],
            axis=1,
            inplace=True,
        )

        return employee_data_joined

    except Exception as e:
        logging.error("Error in preprocessing employee data: %s", e)
        raise


def get_employee_timesheet_data(in_time_data, out_time_data):
    """
    Process in-time and out-time data to calculate daily hours worked and related metrics.
    """
    logging.info("Processing timesheet employee data...")
    try:
        # Process in-time data
        in_time_data.rename(
            columns={in_time_data.columns[0]: "EmployeeID"}, inplace=True
        )
        for col in in_time_data.columns[1:]:
            in_time_data[col] = pd.to_datetime(in_time_data[col], errors="coerce")

        in_time_data_melted = in_time_data.melt(
            id_vars=["EmployeeID"], var_name="date", value_name="entry_time"
        )
        # Process out-time data
        out_time_data.rename(
            columns={out_time_data.columns[0]: "EmployeeID"}, inplace=True
        )
        for col in out_time_data.columns[1:]:
            out_time_data[col] = pd.to_datetime(out_time_data[col], errors="coerce")

        out_time_data_melted = out_time_data.melt(
            id_vars="EmployeeID", var_name="date", value_name="exit_time"
        )

        in_time_data_melted.set_index(["EmployeeID", "date"], inplace=True)
        out_time_data_melted.set_index(["EmployeeID", "date"], inplace=True)

        # Merge in-time and out-time data
        time_data_employees = in_time_data_melted.join(
            out_time_data_melted, how="left", lsuffix="_in", rsuffix="_out"
        )
        # Fill missing values and calculate daily hours worked
        time_data_employees["entry_time"] = time_data_employees["entry_time"].fillna(
            pd.Timestamp("00:00:00")
        )
        time_data_employees["exit_time"] = time_data_employees["exit_time"].fillna(
            pd.Timestamp("00:00:00")
        )
        time_data_employees["entry_time"] = time_data_employees["entry_time"].dt.time
        time_data_employees["exit_time"] = time_data_employees["exit_time"].dt.time

        time_data_employees["daily_hours_worked"] = (
            pd.to_timedelta(time_data_employees["exit_time"].astype(str))
            - pd.to_timedelta(time_data_employees["entry_time"].astype(str))
        ).dt.total_seconds() / 3600
        # Calculate absences, mean daily worked hours, overtime, and early departures
        total_absents = time_data_employees["daily_hours_worked"] == 0

        yearly_absents = (
            time_data_employees[total_absents]
            .groupby("EmployeeID")["daily_hours_worked"]
            .count()
        )
        yearly_absents = yearly_absents.reset_index()
        yearly_absents = yearly_absents.rename(
            columns={"daily_hours_worked": "yearly_absents"}
        )

        mean_daily_worked_hours = (
            time_data_employees[~total_absents]
            .groupby("EmployeeID")["daily_hours_worked"]
            .mean()
        )
        mean_daily_worked_hours = mean_daily_worked_hours.reset_index()
        mean_daily_worked_hours = mean_daily_worked_hours.rename(
            columns={"daily_hours_worked": "mean_daily_worked_hours"}
        )

        average_entry_time = (
            time_data_employees[~total_absents]
            .groupby("EmployeeID")["entry_time"]
            .apply(
                lambda x: np.mean([t.hour * 3600 + t.minute * 60 + t.second for t in x])
                / 3600
            )
        )
        average_entry_time = average_entry_time.reset_index()
        average_entry_time = average_entry_time.rename(
            columns={"entry_time": "average_entry_time"}
        )

        employees_time_data = mean_daily_worked_hours.join(
            yearly_absents.set_index("EmployeeID"), on="EmployeeID", how="left"
        )
        employees_time_data = employees_time_data.join(
            average_entry_time.set_index("EmployeeID"), on="EmployeeID", how="left"
        )

        return employees_time_data

    except Exception as e:
        logging.error("Error in processing employee timesheet data: %s", e)
        raise


def save_clean_data(employee_data_joined, employees_time_data, file_name):
    """
    Save cleaned and processed employee data to a CSV file.
    """
    try:
        employee_data = employee_data_joined.join(
            employees_time_data.set_index("EmployeeID"), on="EmployeeID", how="left"
        )
        clean_employee_data = employee_data.dropna()
        clean_employee_data = clean_employee_data.reset_index()
        clean_employee_data = clean_employee_data.drop(columns=["EmployeeID"])
        logging.info(clean_employee_data.info())
        clean_employee_data.to_csv(file_name, index=False)
        logging.info(f"Successfully saved preprocessed data to {file_name}")
    except Exception as e:
        logging.error("Error in saving clean data: %s", e)
        raise
