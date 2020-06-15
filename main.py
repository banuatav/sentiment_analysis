import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer


def create_date_var(data):
    review_from = data["from_and_date"]
    review_month = []
    review_year = []

    for i in review_from:
        temp = i.split("review ")[1]
        month = temp.split(" ")[0]
        temp = temp.split(" ")[1]
        year = temp[0:4]
        review_month.append(month)
        review_year.append(year)

    data["review_month "] = review_month
    data["review_year"] = review_year

    return data


def clean_text_data(data):
    reviews = data["review"]
    cleaned_reviews = []

    for text in reviews:
        text = text.lower()

        cleaned_reviews.append(text)

    data["cleaned_reviews"] = cleaned_reviews
    return data

def sentiment_analysis_par(paragraph):

    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(paragraph)
    scores_list = []
    for k in sorted(ss):
        scores_list.append(ss[k])

    return scores_list


def sentiment_analysis_loop(data):
    col_names = ["compound", "neg", "neu", "pos"]

    data_sent = pd.DataFrame(columns=col_names)
    reviews = data["cleaned_reviews"]

    for paragraph in reviews:
        out = sentiment_analysis_par(paragraph)
        out_series = pd.Series(out, index=data_sent.columns)
        data_sent = data_sent.append(out_series, ignore_index=True)
    data = pd.concat([data.reset_index(), data_sent.reset_index()], axis=1)
    return data


if __name__ == "__main__":

    raw_data = pd.read_excel("data.xlsx", sheet_name="LDN01-CG")
    raw_data = raw_data[
        [
            "link",
            "from_and_date1",
            "review1",
            "from_and_date2",
            "review2",
            "from_and_date3",
            "review3",
            "from_and_date4",
            "review4",
            "from_and_date5",
            "review5",
        ]
    ]

    data = raw_data[["link", "from_and_date1", "review1"]]
    data = data.rename(
        columns={"link": "link", "from_and_date1": "from_and_date",
        "review1": "review"}, errors="raise",
    )

    data2 = raw_data[["link", "from_and_date2", "review2"]]
    data2 = data2.rename(
        columns={"link": "link", "from_and_date2": "from_and_date",
        "review2": "review"}, errors="raise",
    )

    data3 = raw_data[["link", "from_and_date3", "review3"]]
    data3 = data3.rename(
        columns={"link": "link", "from_and_date3": "from_and_date",
        "review3": "review"}, errors="raise",
    )

    data4 = raw_data[["link", "from_and_date4", "review4"]]
    data4 = data4.rename(
        columns={"link": "link", "from_and_date4": "from_and_date",
        "review4": "review"}, errors="raise",
    )

    data5 = raw_data[["link", "from_and_date5", "review5"]]
    data5 = data5.rename(
        columns={"link": "link", "from_and_date5": "from_and_date",
        "review5": "review"}, errors="raise",
    )

    data = data.append(data2)
    data = data.append(data3)
    data = data.append(data4)
    data = data.append(data5)

    data = data.dropna()

    data = create_date_var(data)
    data = clean_text_data(data)
    
    data = sentiment_analysis_loop(data)
    print(data.head())
    