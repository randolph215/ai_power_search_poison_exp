import pandas as pd

df = pd.read_parquet("datasets/rephrase_data_samples_en_table.parquet")

# print(df.columns.tolist())

columns_of_interest = ['input_query_id','input_product_description', 'label_product_title']
# print(df[columns_of_interest].head(10))
first_row = df.loc[0, columns_of_interest]

with pd.option_context('display.max_colwidth', None):
    print(first_row)