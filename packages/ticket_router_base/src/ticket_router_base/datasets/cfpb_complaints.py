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
            "sub_issue",
            "Sub-issue",
            [],  # populated dynamically in load() from data
        ),
    ]
    generation_task = GenerationTask("company_response", "Company response to consumer")
    discrete_feature_columns = [
        "State",
        "ZIP code",
        "Tags",
        "Submitted via",
        "Company",
    ]

    def load(self):
        """Override to filter out rows with empty narrative, sample, and infer sub-issue labels."""
        import pandas as pd

        df = pd.read_csv(
            self.csv_path,
            delimiter=self.delimiter,
            encoding=self.encoding,
            nrows=10000,  # dev sampling; remove for full training
        )
        df = df[df[self.body_column].notna()].copy()

        # Infer sub-issue labels from sampled data and replace rare ones with "Other"
        sub_issue_col = "Sub-issue"
        if sub_issue_col in df.columns:
            # keep rows where sub-issue is known
            df = df[df[sub_issue_col].notna()].copy()
            vc = df[sub_issue_col].value_counts()
            # keep sub-issues that appear at least 5 times; map others to "Other"
            frequent = set(vc[vc >= 5].index)
            df[sub_issue_col] = df[sub_issue_col].apply(
                lambda x: x if x in frequent else "Other"
            )
            # update the classification task labels dynamically
            for task in self.classification_tasks:
                if task.name == "sub_issue":
                    # dataclass is frozen, so we replace the whole task list
                    new_tasks = []
                    for t in self.classification_tasks:
                        if t.name == "sub_issue":
                            labels = sorted(df[sub_issue_col].unique())
                            new_tasks.append(
                                ClassificationTask(t.name, t.target_column, labels)
                            )
                        else:
                            new_tasks.append(t)
                    object.__setattr__(self, "classification_tasks", new_tasks)
                    break

        return self._df_to_records(df)
