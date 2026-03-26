from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Feature specification (đặc tả biến) cho pipeline."""

    id_columns: tuple[str, ...] = ("EmployeeNumber", "EmployeeCount", "Over18", "StandardHours", "Department")
    target_column: str = "Attrition"
    categorical_columns: tuple[str, ...] = (
        "BusinessTravel", "EducationField", 
        "Gender", "JobRole", "MaritalStatus", "OverTime"
    )
    numeric_columns: tuple[str, ...] = (
        "Age", "DailyRate", "DistanceFromHome", "Education", 
        "EnvironmentSatisfaction", "HourlyRate", "JobInvolvement", 
        "JobLevel", "JobSatisfaction", "MonthlyIncome", "MonthlyRate", 
        "NumCompaniesWorked", "PercentSalaryHike", "PerformanceRating", 
        "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", 
        "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", 
        "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"
    )
