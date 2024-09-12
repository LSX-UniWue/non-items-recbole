import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd

def read_csv(dataset_dir,
             file: str,
             file_type: str,
             sep: str,
             header: bool = None,
             encoding: str = "utf-8"
             ) -> pd.DataFrame:
    file_path = dataset_dir +"/"+ f"{file}{file_type}"
    return pd.read_csv(file_path, sep=sep, header=header, engine="python", encoding=encoding)

class Movielens20MConverter():
    RATING_USER_COLUMN_NAME = 'userId'
    RATING_MOVIE_COLUMN_NAME = 'movieId'
    RATING_TIMESTAMP_COLUMN_NAME = 'timestamp'

    def __init__(self, delimiter="\t", min_item_feedback=4, min_sequence_length=4):
        self.delimiter = delimiter
        self.min_item_feedback = min_item_feedback
        self.min_sequence_length = min_sequence_length

    def apply(self, input_dir, output_file):
        file_type = ".csv"
        header = 0
        sep = ","
        name = "ml-20m"
        location = input_dir+"/"+name

        ratings_df = read_csv(location, "ratings", file_type, sep, header)

        movies_df = read_csv(location, "movies", file_type, sep, header)

        links_df = read_csv(location, "links", file_type, sep, header)
        ratings_df = pd.merge(ratings_df, links_df)

        merged_df = pd.merge(ratings_df, movies_df).sort_values(
            by=[Movielens20MConverter.RATING_USER_COLUMN_NAME, Movielens20MConverter.RATING_TIMESTAMP_COLUMN_NAME])

        # Remove unnecessary columns, we keep movieId here so that we can filter later.
        merged_df = merged_df.drop('imdbId', axis=1).drop('tmdbId', axis=1)

        train, val, test = self._split(merged_df)
        output_file = Path(output_file)
        os.makedirs(output_file, exist_ok=True)

        train.to_csv(path_or_buf=os.path.join(output_file, name + '.train.csv'), sep=self.delimiter, index=False)
        val.to_csv(path_or_buf=os.path.join(output_file, name + '.validation.csv'), sep=self.delimiter,
                          index=False)
        test.to_csv(path_or_buf=os.path.join(output_file, name + '.test.csv'), sep=self.delimiter, index=False)


    def _apply_min_item_feedback(self, page_views):
        aggregated = page_views.groupby(['movieId']).size()
        filtered = aggregated.apply(lambda v: v >= self.min_item_feedback)
        filtered = filtered.reset_index()
        filtered.columns = ['movieId', 'item_feedback_bool']
        ids = filtered[filtered['item_feedback_bool'] == False]['movieId'].tolist()
        full_dataset = page_views[~page_views['movieId'].isin(ids)].copy()
        return full_dataset

    def _apply_min_sequence_length(self, dataset):
        aggregated = dataset.groupby(['userId']).size()
        filtered = aggregated.apply(lambda v: v >= self.min_sequence_length)
        filtered = filtered.reset_index()
        filtered.columns = ['userId', 'min_sequence_bool']
        ids = filtered[filtered['min_sequence_bool']]['userId'].tolist()
        dataset = dataset[dataset['userId'].isin(ids)].copy()
        return dataset

    def _split(self, pd_data, train_percent: float = 0.8, val_percent: float = 0.1, test_percent: float = 0.1):
        if not (train_percent + val_percent + test_percent == 1.0):
            raise ValueError("Train, validation, and test percentages must sum to 1.")
        n = len(pd_data)
        train_end = int(train_percent * n)
        val_end = train_end + int(val_percent * n)
        train = pd_data.iloc[:train_end]
        val = pd_data.iloc[train_end:val_end]
        test = pd_data.iloc[val_end:]

        train = self._apply_min_sequence_length(train)
        validation = self._apply_min_sequence_length(val)
        test = self._apply_min_sequence_length(test)

        train.sort_values(['timestamp'], inplace=True)
        validation.sort_values(['timestamp'], inplace=True)
        test.sort_values(['timestamp'], inplace=True)
        return train, validation, test




class CoveoConverter:

    def __init__(self, end_of_train, end_of_validation, min_item_feedback, min_sequence_length, include_pageviews,
                 prefix, search_sessions_only=False, include_search=False, filter_immediate_duplicates=False,
                 delimiter: str = "\t"):
        self.end_of_train = end_of_train
        self.end_of_validation = end_of_validation
        self.min_item_feedback = min_item_feedback
        self.min_sequence_length = min_sequence_length
        self.include_pageviews = include_pageviews
        self.delimiter = delimiter
        self.prefix = prefix
        self.search_list_page = include_search
        self.search_sessions_only = search_sessions_only
        self.filter_immediate_duplicates = filter_immediate_duplicates

    def apply(self, input_dir: Path, output_file: Path):
        output_dir = output_file.parent
        browsing_train, search_train, sku_to_content = self._load_raw_files(input_dir)

        self._convert_vectors_to_lists(search_train)
        self._convert_vectors_to_arrays(sku_to_content)

        self._remove_duplicates(browsing_train)
        # browsing_train = self._add_search_clicks(browsing_train, search_train)
        browsing_train, page_views = self._handle_pageviews(browsing_train)

        # Filter immediate duplicates
        if self.filter_immediate_duplicates:
            browsing_train["duplicate"] = browsing_train.sort_values(['server_timestamp_epoch_ms']).groupby(["session_id_hash"])["product_sku_hash"].shift()
            browsing_train = browsing_train[browsing_train["duplicate"] != browsing_train["product_sku_hash"]].drop(columns="duplicate")

        product_dataset = self._merge_tables(browsing_train, sku_to_content)
        product_dataset = self._apply_min_item_feedback(product_dataset)
        page_views = self._apply_min_page_view_feedback(page_views)
        #self._fill_nan_values(product_dataset)

        desc_vector_dict = None #self._create_desc_vector_dict(sku_to_content)
        img_vector_dict = None #self._create_img_vector_dict(sku_to_content)

        search_train = self._prepare_search_list_pages(search_train, sku_to_content)

        product_dataset["item_id_type"] = 1
        test, train, validation = self._create_split(product_dataset, search=search_train, page_views=page_views)

        if "query_vector" in test.columns:
            nan_vector = np.zeros(45,dtype=float).tolist()
            def replace_nan_with_predefined_list(row):
                if pd.isna(row['query_vector']):
                    row['query_vector'] = nan_vector
                return row

            test = test.apply(replace_nan_with_predefined_list, axis=1)
            train = train.apply(replace_nan_with_predefined_list, axis=1)
            validation = validation.apply(replace_nan_with_predefined_list, axis=1)
        self._fill_nan_values(test)
        self._fill_nan_values(train)
        self._fill_nan_values(validation)

        if not os.path.exists(output_file):
            output_file.parent.mkdir(parents=True, exist_ok=True)
        self._export_files(desc_vector_dict, img_vector_dict, output_dir, test, train, validation, prefix=self.prefix)

    def _prepare_search_list_pages(self, search_clicks, sku_to_content):
        search_clicks = search_clicks[
            ['session_id_hash', 'product_skus_hash', 'server_timestamp_epoch_ms', 'query_vector']].copy()
        search_clicks.drop_duplicates(inplace=True)
        search_clicks = search_clicks[search_clicks['product_skus_hash'].notnull()]
        search_clicks["product_skus_hash"] = search_clicks["product_skus_hash"].str.replace('\[|\]|\'', '')
        search_clicks["product_skus_hash"] = search_clicks["product_skus_hash"].str.split(",")

        #Get most common cats
        search_clicks_expl = search_clicks.explode("product_skus_hash")
        category_dict = sku_to_content[["product_sku_hash", "category_hash"]].set_index("product_sku_hash").to_dict()[
            "category_hash"]
        search_category = search_clicks_expl[
            ["session_id_hash", "server_timestamp_epoch_ms", 'product_skus_hash']].copy()
        search_category["category_hash"] = search_category["product_skus_hash"].map(category_dict)
        search_category.dropna()
        search_category.assign(category_hash=search_category['category_hash'].str.split('/')).explode('category_hash')
        search_category = self.get_list_page_categories(search_category)

        #First result
        search_clicks["first_result_product"] = search_clicks["product_skus_hash"].apply(lambda x: x[0])
        search_clicks["first_result_cat"] = search_clicks["first_result_product"].map(category_dict)
        search_clicks["first_result_category_hash"] = search_clicks["first_result_cat"]
        search_clicks = search_clicks[["session_id_hash", "server_timestamp_epoch_ms", 'query_vector',
                                       'first_result_product','first_result_cat']]
        search_clicks = search_clicks.merge(search_category, how="left", on=["session_id_hash", "server_timestamp_epoch_ms"])
        search_clicks["item_id_type"] = 0
        search_clicks["category_product_id"] = search_clicks["category_hash"]
        search_clicks["event_type"] = "search"
        search_clicks["product_action"] = "search"
        search_clicks["product_sku_hash"] = "SEARCHLIST"
        search_clicks["hashed_url"] = "SEARCHLIST"
        search_clicks["price_bucket"] = "0"
        search_clicks['server_timestamp_epoch_ms'] = search_clicks[['server_timestamp_epoch_ms']].astype(int)

        return search_clicks

    def get_list_page_categories(self, df):
        def concat_values(df):
            return df['category_hash'].str.cat(sep='/')

        counts = df.groupby(['session_id_hash', 'server_timestamp_epoch_ms'])[
            'category_hash'].value_counts().reset_index(
            name='count')
        counts = counts.groupby(["session_id_hash", "server_timestamp_epoch_ms"]).apply(
            lambda x: x.nlargest(5, 'count')).reset_index(drop=True)
        result = counts.groupby(["session_id_hash", "server_timestamp_epoch_ms"]).apply(concat_values).reset_index(
            name='category_hash')
        return result

    def _load_raw_files(self, input_dir):
        browsing_train = pd.read_csv(os.path.join(input_dir, "browsing_train.csv"), header=0)
        search_train = pd.read_csv(os.path.join(input_dir, "search_train.csv"), header=0)
        sku_to_content = pd.read_csv(os.path.join(input_dir, "sku_to_content.csv"), header=0)
        return browsing_train, search_train, sku_to_content

    def _convert_vectors_to_lists(self, search_train):
        search_train['clicked_skus_hash'] = search_train['clicked_skus_hash'].apply(self._convert_str_to_list)

    def _convert_str_to_list(self, x):
        if pd.isnull(x):
            return x
        return ast.literal_eval(x)

    def _convert_vectors_to_arrays(self, sku_to_content):
        sku_to_content['description_vector'] = sku_to_content['description_vector'].apply(self._convert_str_to_pdarray)
        sku_to_content['image_vector'] = sku_to_content['image_vector'].apply(self._convert_str_to_pdarray)

    def _convert_str_to_pdarray(self, x):
        if pd.isnull(x):
            return x
        list_x = ast.literal_eval(x)
        return pd.array(data=list_x, dtype=float)

    def _remove_duplicates(self, browsing_train):
        browsing_train.drop_duplicates(inplace=True)
        # Remove indices of 'pageview' interactions from duplicated events where an interaction generate a detail and a pageview event
        tmp = browsing_train[(browsing_train.event_type == 'pageview') & (
            browsing_train.duplicated(['session_id_hash', 'server_timestamp_epoch_ms'], keep="first"))]
        browsing_train.drop(tmp.index, inplace=True)
        tmp2 = browsing_train[(browsing_train.event_type == 'pageview') & (
            browsing_train.duplicated(['session_id_hash', 'server_timestamp_epoch_ms'], keep="last"))]
        browsing_train.drop(tmp2.index, inplace=True)

    def _add_search_clicks(self, browsing_train, search_train):
        search_clicks = self._extract_search_clicks(search_train)
        browsing_train = pd.concat([browsing_train, search_clicks])
        return browsing_train

    def _extract_search_clicks(self, search_train):
        search_clicks = search_train[['session_id_hash', 'clicked_skus_hash', 'server_timestamp_epoch_ms']].copy()
        search_clicks['event_type'] = 'event_product'
        search_clicks['product_action'] = 'search'
        search_clicks = search_clicks[search_clicks['clicked_skus_hash'].notnull()]
        search_clicks = self._unstack_list_of_clicked_items_to_multiple_rows(search_clicks)
        search_clicks['hashed_url'] = search_clicks['product_sku_hash']
        # duplicates could indicate interest but are removed here
        search_clicks.drop_duplicates(inplace=True)
        return search_clicks

    def _unstack_list_of_clicked_items_to_multiple_rows(self, search_clicks):
        lst_col = 'clicked_skus_hash'
        search_clicks = pd.DataFrame({
            col: np.repeat(search_clicks[col].values, search_clicks[lst_col].str.len()) for col in
            search_clicks.columns.difference([lst_col])}).assign(
            **{lst_col: np.concatenate(search_clicks[lst_col].values)})[search_clicks.columns.tolist()]
        search_clicks.columns = ['session_id_hash', 'product_sku_hash', 'server_timestamp_epoch_ms',
                                 'event_type', 'product_action']
        return search_clicks

    def _handle_pageviews(self, browsing_train):
        product_views = browsing_train[browsing_train['product_sku_hash'].notnull()]
        browsing_train['category_product_id'] = browsing_train['product_sku_hash']
        #browsing_train['product_sku_hash'] = browsing_train['product_sku_hash'].fillna(browsing_train['pageview'])
        browsing_train['category_product_id'] = browsing_train['category_product_id'].fillna(browsing_train['hashed_url'])
        page_views = browsing_train[browsing_train.event_type == 'pageview']
        page_views["item_id_type"] = 0
        return (product_views, page_views)

    def _merge_tables(self, browsing_train, sku_to_content):
        full_dataset = pd.merge(browsing_train, sku_to_content, on='product_sku_hash', how='left')
        full_dataset.drop(columns=['description_vector', 'image_vector'], inplace=True)
        full_dataset.sort_values(['session_id_hash', 'server_timestamp_epoch_ms'], inplace=True)
        return full_dataset

    def _apply_min_item_feedback(self, full_dataset):
        aggregated = full_dataset[full_dataset['event_type'] == 'event_product'].groupby(['product_sku_hash']).size()
        filtered = aggregated.apply(lambda v: v >= self.min_item_feedback)
        filtered = filtered.reset_index()
        filtered.columns = ['product_sku_hash', 'item_feedback_bool']
        ids = filtered[filtered['item_feedback_bool'] == False]['product_sku_hash'].tolist()
        full_dataset = full_dataset[~full_dataset['product_sku_hash'].isin(ids)].copy()
        return full_dataset

    def _apply_min_page_view_feedback(self, page_views):
        aggregated = page_views.groupby(['hashed_url']).size()
        filtered = aggregated.apply(lambda v: v >= self.min_item_feedback)
        filtered = filtered.reset_index()
        filtered.columns = ['hashed_url', 'item_feedback_bool']
        ids = filtered[filtered['item_feedback_bool'] == False]['hashed_url'].tolist()
        full_dataset = page_views[~page_views['hashed_url'].isin(ids)].copy()
        return full_dataset

    def _fill_nan_values(self, full_dataset):
        full_dataset.fillna(value={"product_action": "view", "price_bucket": "missing", "category_hash": "missing",
                                   "first_result_product":"missing", "first_result_cat":"missing"},
                            inplace=True)

    def _create_desc_vector_dict(self, sku_to_content):
        desc_vector_dict = sku_to_content[['product_sku_hash', 'description_vector']]
        desc_vector_dict = desc_vector_dict[desc_vector_dict['description_vector'].notnull()]
        return desc_vector_dict

    def _create_img_vector_dict(self, sku_to_content):
        img_vector_dict = sku_to_content[['product_sku_hash', 'image_vector']]
        img_vector_dict = img_vector_dict[img_vector_dict['image_vector'].notnull()]
        return img_vector_dict

    def _create_split(self, full_dataset, search, page_views):

        full_dataset["category_product_id"] = full_dataset["product_sku_hash"]
        full_dataset["first_result_product"] = full_dataset["product_sku_hash"]
        full_dataset["first_result_cat"] = full_dataset["product_sku_hash"]
        full_dataset.sort_values(['server_timestamp_epoch_ms'], inplace=True)
        train = full_dataset.loc[(full_dataset['server_timestamp_epoch_ms'] <= self.end_of_train)].copy()
        validation = full_dataset.loc[
            (full_dataset['server_timestamp_epoch_ms'] <= self.end_of_validation) & (
                    full_dataset['server_timestamp_epoch_ms'] > self.end_of_train)].copy()
        test = full_dataset.loc[(full_dataset['server_timestamp_epoch_ms'] > self.end_of_validation)].copy()

        train = self._apply_min_sequence_length(full_dataset)
        validation = self._apply_min_sequence_length(full_dataset)
        test = self._apply_min_sequence_length(full_dataset)

        page_views["category_product_id"] = page_views["hashed_url"]
        page_views["product_sku_hash"] = "PAGE_VIEW"
        page_views_train = page_views.loc[(page_views['server_timestamp_epoch_ms'] <= self.end_of_train)].copy()
        page_views_validation = page_views.loc[
            (page_views['server_timestamp_epoch_ms'] <= self.end_of_validation) & (
                    page_views['server_timestamp_epoch_ms'] > self.end_of_train)].copy()
        page_views_test = page_views.loc[(page_views['server_timestamp_epoch_ms'] > self.end_of_validation)].copy()

        search["category_product_id"] = search["category_hash"]
        search_train = search.loc[(search['server_timestamp_epoch_ms'] <= self.end_of_train)].copy()
        search_validation = search.loc[
            (search['server_timestamp_epoch_ms'] <= self.end_of_validation) & (
                    search['server_timestamp_epoch_ms'] > self.end_of_train)].copy()
        search_test = search.loc[(search['server_timestamp_epoch_ms'] > self.end_of_validation)].copy()

        # Add search pages if necessary
        train = self._filtered_concat(train, search_train, page_views_train)
        validation = self._filtered_concat(validation, search_validation, page_views_validation)
        test = self._filtered_concat(test, search_test, page_views_test)

        train.sort_values(['session_id_hash', 'server_timestamp_epoch_ms'], inplace=True)
        validation.sort_values(['session_id_hash', 'server_timestamp_epoch_ms'], inplace=True)
        test.sort_values(['session_id_hash', 'server_timestamp_epoch_ms'], inplace=True)

        return test, train, validation

    def _filtered_concat(self, dataframe, search, page_views):
        search = search[search['session_id_hash'].isin(dataframe['session_id_hash'])]
        page_views = page_views[page_views['session_id_hash'].isin(dataframe['session_id_hash'])]
        if self.search_sessions_only:
            dataframe = dataframe[dataframe['session_id_hash'].isin(search['session_id_hash'])]
        if self.search_list_page:
            dataframe = pd.concat([dataframe, search])
        if self.include_pageviews:
            dataframe = pd.concat([dataframe, page_views])
        return dataframe

    def _apply_min_sequence_length(self, dataset):
        aggregated = dataset.groupby(['session_id_hash']).size()
        filtered = aggregated.apply(lambda v: v >= self.min_sequence_length)
        filtered = filtered.reset_index()
        filtered.columns = ['session_id_hash', 'min_sequence_bool']
        ids = filtered[filtered['min_sequence_bool']]['session_id_hash'].tolist()
        dataset = dataset[dataset['session_id_hash'].isin(ids)].copy()
        return dataset

    def _export_files(self, desc_vector_dict, img_vector_dict, output_dir, test, train, validation, prefix="coveo"):
        os.makedirs(output_dir, exist_ok=True)

        train.to_csv(path_or_buf=os.path.join(output_dir, prefix + '.train.csv'), sep=self.delimiter, index=False)
        validation.to_csv(path_or_buf=os.path.join(output_dir, prefix + '.validation.csv'), sep=self.delimiter,
                          index=False)
        test.to_csv(path_or_buf=os.path.join(output_dir, prefix + '.test.csv'), sep=self.delimiter, index=False)
        # desc_vector_dict.to_csv(path_or_buf=os.path.join(output_dir, "desc_vector_dict.csv"), sep=self.delimiter,
        #                        index=False)
        # img_vector_dict.to_csv(path_or_buf=os.path.join(output_dir, "img_vector_dict.csv"), sep=self.delimiter,
        #