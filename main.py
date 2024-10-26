import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sqlalchemy import create_engine

# Establish database connection
engine = create_engine(
    'postgresql://user:password@10.1.133.239:5432/data_query_db')

# Step 1: Load data from the database
query = """
    SELECT o.id AS opportunity_id, p.id AS product_id
    FROM opportunities o
    JOIN opportunity_products op ON o.id = op.opportunity_id
    JOIN products p ON op.product_id = p.id
"""


df = pd.read_sql(query, engine)

# Step 2: Prepare the data for association analysis
# Create a transaction matrix with each row as an opportunity and each column as a product
transaction_df = df.pivot_table(
    index='opportunity_id', columns='product_id', aggfunc=lambda x: 1, fill_value=0)

# Step 3: Apply Apriori Algorithm to find frequently bought together items
# We set a minimum support threshold to focus on meaningful pairs
frequent_itemsets = apriori(
    transaction_df, min_support=0.05, use_colnames=True)
print(frequent_itemsets)
print(transaction_df.head())

# Step 4: Generate association rules to find "people who bought this also bought that"
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# Filter rules to find strong recommendations
rules = rules[(rules['confidence'] > 0.5) & (rules['lift'] > 1)]
print("Rules")
print(rules)
# Step 5: Create recommendation function


def recommend_products(product_id, rules_df):
    """Recommend products based on association rules for a given product_id"""
    recommendations = rules_df[rules_df['antecedents']
                               == frozenset([product_id])]
    return recommendations[['consequents', 'confidence', 'lift']]


def recommend_products(product_id, rules_df):
    """Recommend products based on association rules for a given product_id"""
    product_set = frozenset([product_id])

    # Find recommendations based on antecedents
    recommendations_from_antecedents = rules_df[rules_df['antecedents'] == product_set]

    # Find recommendations based on consequents
    recommendations_from_consequents = rules_df[rules_df['consequents'] == product_set]

    # Combine both recommendations
    recommendations = pd.concat([
        recommendations_from_antecedents[[
            'consequents', 'confidence', 'lift']],
        recommendations_from_consequents[['antecedents', 'confidence', 'lift']]
    ])

    recommendations.reset_index(drop=True, inplace=True)

    # Get the recommended products
    recommended_products = recommendations['consequents'].dropna()

    if not recommended_products.empty:
        for product in recommended_products:
            print(f"Recommended Product: {product}")
    else:
        print("No recommendations available for this product.")


# Example usage:
# Recommend products for a specific product ID
product_id = 101  # Replace with actual product ID
recommended_products = recommend_products(product_id, rules)
print("Recommended")
print(recommended_products)
