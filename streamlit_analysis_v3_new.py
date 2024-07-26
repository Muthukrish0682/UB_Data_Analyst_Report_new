import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

#read input file
df = pd.read_csv(r"fragance_it_keepa.csv")
combined_df = pd.read_csv(r"combined_pricing_v1.csv")

# removing the column "🚚" in the headers
df.columns = [col.replace("🚚", "").strip() for col in df.columns]

# Table of Contents
st.sidebar.header('Table of Contents')
st.sidebar.markdown("""
1. [Initial Observations](#initial-observations)
2. [Brand Analysis](#brand-analysis)
3. [Price Comparison Across Sources](#price-comparision)
""")




# Add Introduction
st.markdown("""
<h1>Welcome to the Amazon Sales Analysis Dashboard</h1>
<p>This dashboard provides a comprehensive analysis of brand performance, Price Comparison Across Sources</p>
<p>This analysis is based on Italy location</p>
            
            """, unsafe_allow_html=True)


# Initial Observations
st.markdown("<a id='initial-observations'></a><h2>Initial Observations:</h2>", unsafe_allow_html=True)
st.markdown("""
<p>We start by examining the key aspects of the data:</p>
<ul>
    <li><b>Product Identification:</b> Unique identifiers like ASIN, EAN, and PartNumber.</li>
    <li><b>Product Details:</b> Product attributes such as Title, Brand, and Image.</li>
    <li><b>Sales Data:</b> Sales ranks and their changes over time.</li>
    <li><b>Pricing Data:</b> Current Buy Box prices and price changes over various periods.</li>
    <li><b>Category:</b> Main, Sub, and Tree categories.</li>
</ul>
""", unsafe_allow_html=True)


sale_rank = df[['ASIN', 'Sales Rank: 30 days avg.', 'Sales Rank: 90 days avg.']]

# Calculate Sales Rank Deviation
df['Sales Rank Deviation'] = sale_rank['Sales Rank: 90 days avg.'] - sale_rank['Sales Rank: 30 days avg.']

# # Check the data with the new Sales Rank Deviation column
# st.write("Data with Sales Rank Deviation:")
# st.write(df.head())

# -----------------------------------------------------------------------------------------------------

st.markdown("<a id='brand-analysis'></a><h2>1. Brand Analysis:</h2>", unsafe_allow_html=True)
# st.title("Brand Analysis:")

# Sort by Sales Rank Deviation in ascending order
sorted_df = df.sort_values(by='Sales Rank Deviation', ascending=False)

# Select top 10 to 15 products
top_products_df = sorted_df.head(15)




# Create bar chart using Plotly
fig = px.bar(
    top_products_df,
    x='ASIN',
    y='Sales Rank Deviation',
    color='Brand',
    title='Top 10 to 15 Products by Sales Rank Deviation',
    labels={'ASIN': 'Product ASIN', 'Sales Rank Deviation': 'Sales Rank Deviation'},
    text='Sales Rank Deviation'
)

# Update layout to sort bars in ascending order
fig.update_layout(xaxis={'categoryorder':'total ascending'})

# Streamlit app
st.header('Top Sales Rank Products by Brand')

# # Display the DataFrame of top products
# st.write("Top 10 to 15 Products by Sales Rank Deviation:")
# st.write(top_products_df[['ASIN', 'Brand', 'Sales Rank Deviation']])

# Display the Plotly chart
st.plotly_chart(fig)


# Focus on High-Deviation Products
with st.expander('5. Focus on High-Deviation Products:'):
    st.write("**Chanel:**")
    st.write("Since Chanel products exhibit the highest sales rank deviations, consider increasing promotional activities and advertising for these products during peak times. This could help capitalize on existing consumer interest and further boost sales.")
    st.write("**Dior and Versace:**")
    st.write("Similarly, products from Dior and Versace with significant deviations should be targeted for promotional campaigns to amplify their already considerable sales fluctuations.")

# ------------------------------------------------------------------------------------------------

# Sort by Sales Rank Deviation in ascending order
sorted_df = df.sort_values(by='Sales Rank Deviation', ascending=True)

# Select top 10 to 15 products
least_products_df = sorted_df.head(15)

# Create bar chart using Plotly
fig = px.bar(
    least_products_df,
    x='ASIN',
    y='Sales Rank Deviation',
    color='Brand',
    title='Least 10 to 15 Products by Sales Rank Deviation',
    labels={'ASIN': 'Product ASIN', 'Sales Rank Deviation': 'Sales Rank Deviation'},
    text='Sales Rank Deviation'
)

# Update layout to sort bars in ascending order
fig.update_layout(xaxis={'categoryorder':'total ascending'})

# Streamlit app
st.header('Least Sales Rank Products by Brand')

# # Display the DataFrame of least performing products
# st.write("Least 10 to 15 Products by Sales Rank Deviation:")
# st.write(least_products_df[['ASIN', 'Brand', 'Sales Rank Deviation']])

# Display the Plotly chart
st.plotly_chart(fig)


# Capitalize on Stability
with st.expander('Capitalize on Stability:'):
    st.write("**GIORGIO ARMANI, Tom Ford, and Chanel:**")
    st.write("Products with low sales rank deviations, such as those from these brands, suggest consistent demand. Leverage this stability in marketing campaigns to highlight the reliability and consistent popularity of these products.")

# Boost Awareness
with st.expander('Boost Awareness:'):
    st.write("**PETZL and ZWILLING:**")
    st.write("For products with stable but lower deviations, consider increasing visibility through enhanced marketing efforts. This includes search engine optimization (SEO), pay-per-click (PPC) advertising, and social media marketing to reach a wider audience.")



# ------------------------------------------------------------------------------------------------


st.markdown("<a id='price-comparision'></a><h2>2. Price Comparison Across Sources:</h2>", unsafe_allow_html=True)

# st.title('Price Comparison Across Sources')

# Merge datasets on 'asin'
merged_df = pd.merge(combined_df, df, left_on='asin', right_on='ASIN', how='inner')


# st.write(merged_df)

def clean_price(price):
    if pd.isna(price):
        return 0
    if isinstance(price, str):
        return float(price.replace('€ ', '').replace(',', '.'))
    return float(price)


# Apply cleaning function to each column
merged_df['Buy Box : Current'] = merged_df['Buy Box : Current'].apply(clean_price)
merged_df['Buy Box : 30 days avg.'] = merged_df['Buy Box : 30 days avg.'].apply(clean_price)
merged_df['Buy Box : 90 days avg.'] = merged_df['Buy Box : 90 days avg.'].apply(clean_price)
merged_df['Buy Box : 180 days avg.'] = merged_df['Buy Box : 180 days avg.'].apply(clean_price)
merged_df['Buy Box : Lowest'] = merged_df['Buy Box : Lowest'].apply(clean_price)
merged_df['Buy Box : Highest'] = merged_df['Buy Box : Highest'].apply(clean_price)
merged_df['price'] = merged_df['price'].apply(clean_price)

merged_df = merged_df.drop(columns=['Unnamed: 7', 'dd', 'link'])


# Rename columns
merged_df = merged_df.rename(columns={
    'price':'Seller Price',
    'Brand_x': 'Seller',
    'Brand_y': 'Brand'
})


# Rename columns
new_column_names = {
    'Buy Box : Current': 'Buy_Box_Current',
    'Buy Box : 30 days avg.': 'Buy_Box_30_days_avg',
    'Buy Box : 90 days avg.': 'Buy_Box_90_days_avg',
    'Buy Box : 180 days avg.': 'Buy_Box_180_days_avg',
    'Sales Rank Deviation': 'Sales_Rank_Deviation',
    'Buy Box: Is FBA': 'Buy_Box_Is_FBA',
    'Buy Box Seller': 'Buy_Box_Seller',
    'Buy Box : Lowest': 'Buy_Box_Lowest',
    'Buy Box : Is Lowest': 'Buy_Box_Is_Lowest',
    'Buy Box : Is Lowest 90 days': 'Buy_Box_Is_Lowest_90_days',
    'Buy Box : Highest': 'Buy_Box_Highest'
}

merged_df.rename(columns=new_column_names, inplace=True)

# Display the updated DataFrame
# st.write(merged_df.head())

# st.header("Price Distribution Across Sellers")

# Create a Plotly box plot to show price distribution for each ASIN across different sources
fig = px.box(
    merged_df,
    x='asin',
    y='Seller Price',
    color='source',
    title='Price Distribution Across Sellers for Each ASIN',
    labels={'Seller Price': 'Price (€)', 'asin': 'Product ASIN'}
)

# Streamlit app
# st.title('Price Distribution Analysis')

# Display the box plot
st.plotly_chart(fig)

# Calculate average price per ASIN
avg_price_asin = merged_df.groupby('asin')['Seller Price'].mean().reset_index()

# ------------------------------------------------------------------------------------------------------

# Get top 15 ASINs by average price
top_15_asins = avg_price_asin.sort_values(by='Seller Price', ascending=False).head(15)['asin']

# Filter merged_df to include only the top 15 ASINs
top_15_df = merged_df[merged_df['asin'].isin(top_15_asins)]


# Create a Plotly box plot to show price distribution for the top 15 ASINs across different sources
fig = px.box(
    top_15_df,
    x='asin',
    y='Seller Price',
    color='source',
    title='Price Distribution Across Sellers for Top 15 ASINs',
    labels={'Seller Price': 'Price (€)', 'asin': 'Product ASIN'}
)

# Streamlit app
# st.title('Price Comparison Across Different Sources Based on ASIN')

# Display the box plot
st.plotly_chart(fig)



# Source information
sources = ['es', 'fr']  # Example sources, adjust based on your data

# Product Price Trends
# st.header('1. Product Price Trends')

for source in sources:
    st.subheader(f'Price Trends for Source: {source}')
    
    with st.expander("High Price Variability"):
        st.write(
            """
            **ASIN B00HT94USS**: This product shows an average price of €309.91, but there is a significant discrepancy with a maximum price of €409.72 and a minimum price of €0.00 in different sources. This high variability may suggest inconsistencies in pricing data or promotions affecting the product’s price significantly. This can impact the product's perceived value and market positioning.
            """
        )
        
    with st.expander("Stable Pricing"):
        st.write(
            """
            **ASIN B07J2RKJ42**: With an average price of €173.56 and consistent pricing across sources, this product demonstrates stability. Stable pricing indicates a well-established market position where price competition is minimal or controlled.
            """
        )

# Pricing Strategy Analysis
# st.header('2. Pricing Strategy Analysis')

for source in sources:
    st.subheader(f'Pricing Strategy for Source: {source}')
    
    with st.expander("Premium Products"):
        st.write(
            """
            **ASIN B0B3V3VTFD**: This product has the highest average price at €357.13, indicating it is positioned as a premium product. The high price may appeal to a niche market or be associated with high-end features or exclusivity.
            """
        )
        
    with st.expander("Competitive Pricing"):
        st.write(
            """
            **ASIN B00T9VZMWW**: Shows a zero average price, which could imply promotional activities or errors in data. If the zero price is accurate, it may be a strategy to attract attention, or the product might be frequently out of stock.
            """
        )

# Price Consistency and Market Segmentation
# st.header('3. Price Consistency and Market Segmentation')

for source in sources:
    st.subheader(f'Price Consistency and Market Segmentation: {source}')
    
    with st.expander("Consistent Price Points"):
        st.write(
            """
            **ASIN B07JCMKVGN**: The price is relatively consistent with a narrow range between the average, median, and maximum prices, suggesting effective price control and less volatility. This could indicate a stable supply chain and predictable market conditions.
            """
        )
        
    with st.expander("Variable Pricing"):
        st.write(
            """
            **ASIN B00QA9HJDQ**: With a high standard deviation and a broad price range, this product might face intense competition or varying pricing strategies across different sellers. This could be due to dynamic pricing models or seasonal promotions.
            """
        )

# Implications for Pricing Strategy
# st.header('4. Implications for Pricing Strategy')

for source in sources:
    st.subheader(f'Implications for Pricing Strategy for Source: {source}')
    
    with st.expander("Price Monitoring and Adjustments"):
        st.write(
            """
            Businesses should monitor products with high price variability like **ASIN B00HT94USS** and assess the reasons behind the fluctuations. Adjustments may be needed to stabilize pricing and ensure competitive positioning.
            """
        )
        
    with st.expander("Pricing for Premium Products"):
        st.write(
            """
            Products with high average prices, such as **ASIN B0B3V3VTFD**, should be marketed as premium items. Ensure that the product's features and quality justify the high price point to maintain customer perception of value.
            """
        )
        
    with st.expander("Managing Competitive Pricing"):
        st.write(
            """
            For products with competitive pricing, like **ASIN B00QA9HJDQ**, consider strategies to differentiate the product, such as enhancing value through additional features or better customer service, to reduce the impact of price competition.
            """
        )

# ---------------------------------------------------------------------------------------------------------

# Calculate average price per ASIN
avg_price_asin = merged_df.groupby('asin')['Seller Price'].mean().reset_index()

# Get least 15 ASINs by average price
least_15_asins = avg_price_asin.sort_values(by='Seller Price', ascending=True).head(15)['asin']

# Filter merged_df to include only the least 15 ASINs
least_15_df = merged_df[merged_df['asin'].isin(least_15_asins)]

# Create a Plotly box plot to show price distribution for the least 15 ASINs across different sources
fig = px.box(
    least_15_df,
    x='asin',
    y='Seller Price',
    color='source',
    title='Price Distribution Across Sellers for Least 15 ASINs',
    labels={'Seller Price': 'Price (€)', 'asin': 'Product ASIN'}
)

# Streamlit app
# st.title('Price Comparison Across Different Sources Based on ASIN')

# Display the box plot
st.plotly_chart(fig)


# Data for interpretation
data = {
    'source': ['Spain (es)', 'France (fr)'],
    'high_price_variability': {
        'Spain (es)': {
            'asin': 'B00008S0TA',
            'avg_price': 56.23,
            'max_price': 77.62,
            'std_dev': 13.81
        },
        'France (fr)': {
            'asin': 'B007QKT4NG',
            'avg_price': 21.66,
            'max_price': 28.25,
            'std_dev': 4.01
        }
    },
    'stable_pricing': {
        'Spain (es)': {
            'asin': 'B007QKT4NG',
            'avg_price': 18.63,
            'std_dev': 4.01
        },
        'France (fr)': {
            'asin': 'B07CHK6GDY',
            'avg_price': 46.12,
            'std_dev': 9.89
        }
    }
}

# st.title('Price Trends and Insights')

for source in data['source']:
    st.subheader(f'Price Trends Analysis for Source: {source}')
    
    st.expander("Product with High Price Variability").write(
        f"""
        **ASIN {data['high_price_variability'][source]['asin']}**: Exhibits a significant range between its average price (€{data['high_price_variability'][source]['avg_price']:.2f}) and its maximum price (€{data['high_price_variability'][source]['max_price']:.2f}). The standard deviation of €{data['high_price_variability'][source]['std_dev']:.2f} indicates notable variability. This high variability might suggest that this product is frequently on promotion or that there are large differences in pricing strategies among sellers.
        """
    )

    st.expander("Product with Stable Pricing").write(
        f"""
        **ASIN {data['stable_pricing'][source]['asin']}**: Shows relatively stable pricing with an average price of €{data['stable_pricing'][source]['avg_price']:.2f} and a low standard deviation of €{data['stable_pricing'][source]['std_dev']:.2f}. The prices are consistently within a narrow range, indicating a well-established market price.
        """
    )

st.subheader('Pricing Strategy Implications')

st.expander("High Price Variability").write(
    """
    Products with high price variability, such as those identified in Spain and France, may require closer monitoring. Companies should investigate the reasons behind this variability, such as sales promotions, competitive pricing strategies, or seasonal demand fluctuations. High variability can impact customer perception and brand value, so it’s crucial to manage these variations carefully.
    """
)

st.expander("Stable Pricing").write(
    """
    Products with stable pricing should be leveraged for positioning in the market as reliable and consistently priced options. This stability can be advantageous for establishing a strong market position and can reduce the need for frequent price adjustments.
    """
)

st.subheader('Recommendations for Pricing Strategy')

st.expander("For Products with High Variability").write(
    """
    - Implement dynamic pricing strategies to adjust for market fluctuations and competitive pressures.
    - Use pricing analytics to understand the factors driving price changes and adjust marketing strategies accordingly.
    - Monitor competitor pricing and adjust strategies to maintain a competitive advantage.
    """
)

st.expander("For Products with Stable Pricing").write(
    """
    - Continue with the current pricing strategy but keep an eye on market changes that could impact pricing.
    - Emphasize the consistency and reliability of these products in marketing efforts to attract price-sensitive customers.
    - Regularly review pricing to ensure it remains competitive without compromising the perceived value of the product.
    """
)














# ----------------------------------------------------------------------------------------------------------

# # Calculate average price per ASIN
# average_price_per_asin = merged_df.groupby('asin')['Seller Price'].mean().reset_index()

# # Get top 15 ASINs by average price
# top_15_asins = average_price_per_asin.sort_values(by='Seller Price', ascending=False).head(15).round(2)

# # Create bar chart to show average price per ASIN
# fig = px.bar(
#     top_15_asins,
#     x='asin',
#     y='Seller Price',
#     title='Top 15 ASINs by Average Price',
#     labels={'Seller Price': 'Average Price (€)', 'asin': 'Product ASIN'},
#     text='Seller Price'
# )

# # Streamlit app
# # st.title('Top 15 ASINs by Average Price')

# # Display the bar chart
# st.plotly_chart(fig)

# -------------------------------------------------------------------------------------------------------

st.header("Buy Box Price Prediction")

# Streamlit UI
st.header('ASIN Pricing Information')

# User input for ASIN
asin_of_interest = st.text_input('Enter ASIN:', 'B002N5MK3K')

# Filter data for the specific ASIN
asin_df = merged_df[merged_df['asin'] == asin_of_interest]

if not asin_df.empty:
    # Calculate pricing statistics
    suggested_buy_box_price_min = asin_df['Seller Price'].min()
    suggested_buy_box_price_max = asin_df['Seller Price'].max()
    
    # Weighted Average Price
    asin_df['weight'] = asin_df['rating'].fillna(1)
    suggested_buy_box_price_weighted = (asin_df['Seller Price'] * asin_df['weight']).sum() / asin_df['weight'].sum()
    
    # Median Price
    suggested_buy_box_price_median = asin_df['Seller Price'].median()
    
    # Average of Buy Box Prices (excluding current)
    predicted_buy_box_price = asin_df[['Buy_Box_30_days_avg', 'Buy_Box_90_days_avg', 'Buy_Box_180_days_avg']].mean().mean()
    
    # Prepare data for display
    result_data = {
        "Statistic": [
            "Minimum Buy Box Price",
            "Weighted Average Buy Box Price",
            "Median Buy Box Price",
            "Maximum Buy Box Price",
            "Predicted Buy Box Price"
        ],
        "Value (€)": [
            f"€{suggested_buy_box_price_min:.2f}",
            f"€{suggested_buy_box_price_weighted:.2f}",
            f"€{suggested_buy_box_price_median:.2f}",
            f"€{suggested_buy_box_price_max:.2f}",
            f"€{predicted_buy_box_price:.2f}"
        ]
    }

    # Create DataFrame
    result_df = pd.DataFrame(result_data)

    # Highlight "Predicted Buy Box Price"
    def highlight_predicted(s):
        return ['background-color: yellow' if s['Statistic'] == 'Predicted Buy Box Price' else '' for _ in s]

    # Display results in a table with highlight
    st.dataframe(result_df.style.apply(highlight_predicted, axis=1))
else:
    st.write("No data available for the entered ASIN.")

# ---------------------------------------------------------------------------------------------------------------

# Predicted Buy Box Price for all ASIN columns
def calculate_predicted_price(df):
    return df[['Buy_Box_30_days_avg', 'Buy_Box_90_days_avg', 'Buy_Box_180_days_avg']].mean(axis=1)


# Group by 'asin' and calculate the predicted Buy Box price
predicted_df = merged_df.groupby('asin').apply(lambda x: pd.Series({
    'Predicted Buy Box Price': calculate_predicted_price(x).mean().round(2),
    'Buy_Box_Current': x['Buy_Box_Current'].iloc[0]  # Assuming current price is the same for all rows of the same ASIN
})).reset_index()

# Sort by 'Predicted Buy Box Price' and get the top 15 ASINs
top_15_predicted_df = predicted_df.sort_values(by='Predicted Buy Box Price', ascending=False)

# Streamlit UI
st.header('ASINs by Predicted Buy Box Price')

# Display the result
st.dataframe(top_15_predicted_df, use_container_width=True)

# ---------------------------------------------------------------------------------------------------------------
