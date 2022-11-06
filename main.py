import pandas as pd
import argparse
import numpy as np
import csv
import os
from itertools import combinations


def write_output_csv_and_dataframe(output_csv_fullpath: str, itemset_dict: dict, support_dict: dict):
    # init a dataframe
    output_df = pd.DataFrame({'freqset': [], 'support': []})

    with open(output_csv_fullpath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['freqset', 'support'])

        for index in sorted(itemset_dict.keys()):
            df_temp = pd.DataFrame(
                {
                    "support": pd.Series(support_dict[index]),
                    "freqset": pd.Series([frozenset(i) for i in itemset_dict[index]], dtype="object")
                }
            )
            for temp in zip(itemset_dict[index], support_dict[index]):
                # reformat itemsets
                itemsets = ''
                for j in temp[0]:
                    itemsets = f'{itemsets} {j}' if itemsets else str(j)
                writer.writerow(('{' + itemsets + '}', temp[1]))

            output_df = output_df.append(df_temp, ignore_index=True)

    return output_df


def generate_candidates(pre_candidates: np.ndarray):
    pre_items_types = np.unique(pre_candidates.flatten())
    # result = []
    for candidate in pre_candidates:
        # The comparison is performed element-wise and the result of the operation is a Boolean array as desired.
        mask = pre_items_types > candidate[-1]
        valid_items = pre_items_types[mask]

        old_tuple = tuple(candidate)

        for item in valid_items:
            yield from old_tuple
            yield item


def apriori(boolean_pd_dataframe: pd.DataFrame, min_support: float):
    # axis = 0 means along the column
    # Support(A) = Count(A)/Count(ALL_DATA)
    support = np.array(np.count_nonzero(boolean_pd_dataframe.values, axis=0) /
                       boolean_pd_dataframe.values.shape[0]).reshape(-1)  # One shape dimension

    support_dict = {1: support[support >= min_support]}

    # in this case: item index = item name
    np_item_index = np.linspace(
        0, boolean_pd_dataframe.values.shape[1]-1, dtype=int)  # column_number

    itemset_dict = {1: np_item_index[support >= min_support].reshape(-1, 1)}

    max_itemset = 1

    while max_itemset:
        next_max_itemset = max_itemset + 1
        candidates = generate_candidates(itemset_dict[max_itemset])
        candidates = np.fromiter(candidates, dtype=int)
        candidates = candidates.reshape(-1, next_max_itemset)

        if candidates.size == 0:
            break

        support = np.array(
            np.sum(np.all(boolean_pd_dataframe.values[:, candidates], axis=2), axis=0) / boolean_pd_dataframe.values.shape[0]).reshape(-1)

        _mask = (support >= min_support).reshape(-1)  # One shape dimension

        if any(_mask):
            itemset_dict[next_max_itemset] = np.array(candidates[_mask])
            support_dict[next_max_itemset] = np.array(support[_mask])
            max_itemset = next_max_itemset
        else:
            break

    return itemset_dict, support_dict


def association_rule(output_df: pd.DataFrame, min_conf: float):
    # metrics for association rules
    metric_dict = {
        "support": lambda s_xy, _, __: s_xy,
        "confidence": lambda s_xy, s_x, _: s_xy / s_x,
        # Lift(A -> B) = Confidence(A -> B) / P(B) = P(B|A) / P(B)
        "lift": lambda s_xy, s_x, s_y: metric_dict["confidence"](s_xy, s_x, s_y) / s_y
    }

    # get dict of {frequent itemset} -> support
    frozenset_vect = np.vectorize(lambda x: frozenset(x))
    frequent_items_dict = dict(
        zip(frozenset_vect(output_df["freqset"].values), output_df["support"].values))

    rules = []
    rule_supports = []

    # iterate over all frequent itemsets
    for key in frequent_items_dict.keys():
        s_xy = frequent_items_dict[key]
        # to find all possible combinations
        for idx in range(len(key) - 1, 0, -1):
            # of antecedent and consequent
            for c in combinations(key, r=idx):
                antecedent = frozenset(c)
                consequent = key.difference(antecedent)

                try:
                    s_x = frequent_items_dict[antecedent]
                    s_y = frequent_items_dict[consequent]
                except Exception as ex:
                    raise ValueError(ex)

                score = metric_dict["confidence"](s_xy, s_x, s_y)
                if score >= min_conf:
                    rules.append(f'{set(antecedent)} -> {set(consequent)}')
                    rule_supports.append([s_xy, s_x, s_y])

    columns_ordered = [
        "support",
        "confidence",
        "lift"
    ]

    # check if frequent rule was generated
    if not rule_supports:
        return pd.DataFrame(columns=["rule"] + columns_ordered)

    # generate metrics
    rule_supports = np.array(rule_supports).T.astype(float)

    output_df = pd.DataFrame(
        data=rules,
        columns=["rule"],
    )

    for metric in columns_ordered:
        output_df[metric] = metric_dict[metric](
            rule_supports[0], rule_supports[1], rule_supports[2])

    return output_df


def pd_dataset_to_one_hot_encoded_numpy(pd_dataset, transaction_numbers: int, item_names: list):
    boolean_np_array = np.zeros(
        (transaction_numbers, len(item_names)), dtype=bool)

    for index, row in pd_dataset.iterrows():
        boolean_np_array[row['transaction_id'] - 1,
                         item_names.index(row["item"])] = True
    return boolean_np_array


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_sup', type=float,
                        default=0.1, help='Minimum support')

    parser.add_argument('--min_conf', type=float,
                        default=0.1, help='Minimum confidence')

    parser.add_argument('--dataset', type=str, default='ibm-2022.txt',
                        help='Dataset to use, please include the extension')

    args = parser.parse_args()

    if args.min_sup <= 0.0:
        raise ValueError(
            f"`min_support` must be a positive number within the interval `(0, 1]`. Got {args.min_sup}"
        )

    # pandas
    pd_dataset = pd.read_csv(
        args.dataset, names=['customer_id', 'transaction_id', 'item'], delimiter=' ', skipinitialspace=True)

    # find all item names
    item_names = [*set(pd_dataset['item'].tolist())]

    print(f'item_names: {item_names}')

    boolean_np_array = pd_dataset_to_one_hot_encoded_numpy(
        pd_dataset, pd_dataset.iloc[-1]['customer_id'], item_names)

    boolean_pd_dataframe = pd.DataFrame(boolean_np_array, columns=item_names)

    # apriori
    itemset_dict, support_dict = apriori(boolean_pd_dataframe, args.min_sup)

    # write output csv
    print()
    print('-'*10 + 'freqset and support' + '-'*10)
    output_df = write_output_csv_and_dataframe(os.path.join(
        'outputs', f"{args.dataset.split('/')[-1].split('.')[0]}-min_sup-{args.min_sup}.csv"), itemset_dict, support_dict)
    print(output_df)
    print()
    print('-'*10 + 'association_rule' + '-'*10)

    # association_rule
    rule_df = association_rule(output_df, args.min_conf)
    # pd.set_option('display.max_rows', None)
    print(rule_df)
    print()
    print(f'totle rules: {rule_df.values.shape[0]}')
