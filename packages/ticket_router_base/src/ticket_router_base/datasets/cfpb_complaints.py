"""CFPB consumer complaints dataset."""

from ticket_router_base.config import PROJECT_ROOT
from ticket_router_base.datasets.base import (
    BaseDataset,
    ClassificationTask,
    GenerationTask,
)


class CFPBComplaintsDataset(BaseDataset):
    """Consumer Financial Protection Bureau complaints dataset (~25M rows)."""

    name = "cfpb-complaints"
    csv_path = PROJECT_ROOT / "dataset" / "complaints.csv"
    title_column = None
    body_column = "Consumer complaint narrative"
    language_column = None
    id_column = "Complaint ID"

    # label lists are shortened for brevity; full lists should be inferred from data
    classification_tasks = [
        ClassificationTask(
            "product",
            "Product",
            [
                "Bank account or service",
                "Checking or savings account",
                "Consumer Loan",
                "Credit card",
                "Credit card or prepaid card",
                "Credit reporting",
                "Credit reporting, credit repair services, or other personal consumer reports",
                "Debt collection",
                "Money transfer, virtual currency, or money service",
                "Money transfers",
                "Mortgage",
                "Other financial service",
                "Payday loan",
                "Payday loan, title loan, or personal loan",
                "Prepaid card",
                "Student loan",
                "Vehicle loan or lease",
                "Virtual currency",
            ],
        ),
        ClassificationTask(
            "issue",
            "Issue",
            [
                "Account opening, closing, or management",
                "Advertising and marketing, including promotional offers",
                "Applying for a mortgage or refinancing an existing mortgage",
                "Attempts to collect debt not owed",
                "Balance transfer",
                "Billing disputes",
                "Billing statement",
                "Can't contact lender",
                "Can't stop charges to bank account",
                "Closing an account",
                "Closing on a mortgage",
                "Communication tactics",
                "Cont'd attempts collect debt not owed",
                "Credit card protection / Debt protection",
                "Credit decision / Underwriting",
                "Credit line increase/decrease",
                "Credit monitoring or identity protection",
                "Credit reporting company's investigation",
                "Dealing with a lender or servicer",
                "Dealing with my lender or servicer",
                "Delinquent account",
                "Deposits and withdrawals",
                "Disclosure verification of debt",
                "False statements or representation",
                "Fees or interest",
                "Fraud or scam",
                "Getting a line of credit",
                "Getting a loan",
                "Identity theft / Fraud / Embezzlement",
                "Improper contact or sharing of info",
                "Improper use of your report",
                "Incorrect information on your report",
                "Loan modification,collection,foreclosure",
                "Loan servicing, payments, escrow account",
                "Making payments",
                "Managing a line of credit",
                "Managing an account",
                "Money was not available when promised",
                "Opening an account",
                "Other features, terms, or problems",
                "Other service problem",
                "Payment to acct not reflected",
                "Payoff process",
                "Problem adding money",
                "Problem canceling or closing account",
                "Problem fraudulent charges",
                "Problem getting a card or closing an account",
                "Problem getting my card",
                "Problem getting statement",
                "Problem receiving a refund",
                "Problem repaying loan",
                "Problem using a card",
                "Problem using card",
                "Problem with a company's investigation into an existing problem",
                "Problem with a credit reporting company's investigation into an existing problem",
                "Problem with additional add-on products or services",
                "Problem with an overdraft",
                "Problem with cash advance",
                "Problem with credit report or credit score",
                "Problem with customer service",
                "Problem with fees",
                "Problem with fraud alerts or security freezes",
                "Problem with lender or servicer",
                "Problem with overdraft",
                "Problem with product or service terms",
                "Problem with purchase amount",
                "Problem with purchase shown on statement",
                "Problem with rewards",
                "Problem with statement",
                "Problem with written notification about debt",
                "Problems at the end of the loan or lease",
                "Problems when you are unable to pay",
                "Received a loan I didn't apply for",
                "Received a loan you didn't apply for",
                "Received marketing you didn't request",
                "Repaying your loan",
                "Settlement process and costs",
                "Shopping for a line of credit",
                "Struggling to pay mortgage",
                "Struggling to pay your loan",
                "Takes out of my bank account",
                "Trouble during payment process",
                "Unauthorized transactions/trans.issues",
                "Unexpected/Other fees",
                "Unable to get your credit report or credit score",
                "Unsolicited issuance of credit card",
                "Using a debit or ATM card",
                "Vehicle was repossessed or sold the vehicle",
                "Written notification about debt",
            ],
        ),
        ClassificationTask(
            "timely_response",
            "Timely response?",
            ["Yes", "No"],
        ),
        ClassificationTask(
            "consumer_disputed",
            "Consumer disputed?",
            ["Yes", "No", "N/A"],
        ),
    ]
    generation_task = GenerationTask("company_response", "Company response to consumer")
    discrete_feature_columns = [
        "Sub-product",
        "State",
        "ZIP code",
        "Tags",
        "Submitted via",
        "Company",
    ]

    def load(self):
        """Override to filter out rows with empty narrative and sample for dev."""
        import pandas as pd

        df = pd.read_csv(
            self.csv_path,
            delimiter=self.delimiter,
            encoding=self.encoding,
            nrows=10000,  # dev sampling; remove for full training
        )
        df = df[df[self.body_column].notna()].copy()
        return self._df_to_records(df)
