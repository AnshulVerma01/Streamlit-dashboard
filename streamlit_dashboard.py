import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import pointbiserialr
from scipy.stats import chi2_contingency

# Load CSV data
file_path = "redcap_dataset.csv"
df = pd.read_csv(file_path)

# Sidebar with filters
st.sidebar.title("Filter Data")

# Checkbox to include all values for Gender
include_all_values_gender = st.sidebar.checkbox("Include All Values for Gender", False)
selected_gender = None

if not include_all_values_gender:
    selected_gender = st.sidebar.selectbox("Select Gender", df['Gender'].unique())

# Radio buttons for age filter
st.sidebar.subheader("Filter by Age")
age_options = ['All Ages', '0-45', '46-60', 'more than 61']
selected_age = st.sidebar.radio("Select Age Range", age_options)

# Multiselect for creating heatmap
create_heatmap = st.sidebar.checkbox("Heatmap", False)

# New options for heatmap selection
if create_heatmap:
    heatmap_selection = st.sidebar.radio("Select Heatmap Type", ["None", "Numerical", "Categorical and numerical", "Contingency table"])

# Display plots for selected column
columns_to_exclude = ['Gender', 'Do you consent to provide information?', 'Specify your date of birth :', 'QUESTIONNAIRE ID', 'LABID', 'Please select your sex :']

# Filter columns for heatmap selection
filtered_numerical_columns = [col for col in df.select_dtypes(include='number').columns if col not in columns_to_exclude]
filtered_categorical_columns = [col for col in df.select_dtypes(include='object').columns if col not in columns_to_exclude]

selected_categorical_column = None
selected_numerical_columns = None

if create_heatmap and heatmap_selection == "Numerical":
    selected_numerical_columns = st.sidebar.multiselect("Select Numerical Columns for Heatmap", filtered_numerical_columns)

elif create_heatmap and heatmap_selection == "Categorical and numerical":
    selected_categorical_column = st.sidebar.selectbox("Select Categorical Column", filtered_categorical_columns)
    selected_numerical_columns = st.sidebar.multiselect("Select Numerical Columns for Heatmap", filtered_numerical_columns)

elif create_heatmap and heatmap_selection == "Contingency table":
    selected_categorical_column = st.sidebar.selectbox("Select First Categorical Column", filtered_categorical_columns)
    selected_categorical_column_2 = st.sidebar.selectbox("Select Second Categorical Column", filtered_categorical_columns)

# Filter data based on user selection
if include_all_values_gender:
    filtered_data = df
    gender_filtered_data = df  # Added this line for the case when all values for gender are selected
else:
    filtered_data = df[df['Gender'] == selected_gender]
    gender_filtered_data = df[df['Gender'] == selected_gender]

# Apply additional filtering based on selected age range
if selected_age != 'All Ages':
    if selected_age == '0-45':
        filtered_data = filtered_data[filtered_data['Age'] <= 45]
        gender_filtered_data = gender_filtered_data[gender_filtered_data['Age'] <= 45]
    elif selected_age == '46-60':
        filtered_data = filtered_data[(filtered_data['Age'] >= 46) & (filtered_data['Age'] <= 60)]
        gender_filtered_data = gender_filtered_data[(gender_filtered_data['Age'] >= 46) & (gender_filtered_data['Age'] <= 60)]
    else:
        filtered_data = filtered_data[filtered_data['Age'] > 61]
        gender_filtered_data = gender_filtered_data[gender_filtered_data['Age'] > 61]

# Main content area
st.title("Redcap Data Analysis")

# Display the filtered data
st.subheader("Filtered Data")
st.write(filtered_data)

# Function to calculate Cramér's V
def cramers_v(contingency_table):
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))


# Plot heatmap if selected
if create_heatmap:
    if heatmap_selection == "Numerical" and selected_numerical_columns:
        st.subheader("Correlation Heatmap for Numerical Columns")

        # Use the filtered_data DataFrame for creating the heatmap
        correlation_matrix_filtered = filtered_data[selected_numerical_columns].corr()

        # Remove content in square brackets from column names
        column_names_filtered = [col.split('[')[0].strip() for col in correlation_matrix_filtered.columns]

        # Create a heatmap with correlation values
        fig_heatmap_filtered = go.Figure(data=go.Heatmap(
            z=correlation_matrix_filtered.values,
            x=column_names_filtered,  # Use modified column names here
            y=column_names_filtered,  # Use modified column names here
            hoverongaps=False,
            colorscale='magma',
            colorbar=dict(title='Correlation'),
        ))

        # Add text annotations with correlation values
        for i in range(len(column_names_filtered)):
            for j in range(len(column_names_filtered)):
                fig_heatmap_filtered.add_annotation(
                    x=column_names_filtered[j],
                    y=column_names_filtered[i],
                    text=f"{correlation_matrix_filtered.values[i, j]:.2f}",
                    showarrow=False,
                    font=dict(size=16, color='black'),
                )

        fig_heatmap_filtered.update_layout(
            xaxis=dict(showgrid=True, linecolor='black', tickfont=dict(color='black')),
            yaxis=dict(showgrid=True, linecolor='black', tickfont=dict(color='black')),
            hoverlabel=dict(font=dict(color='black')),
        )

        st.plotly_chart(fig_heatmap_filtered)

    elif heatmap_selection == "Categorical and numerical" and selected_categorical_column and selected_numerical_columns:
        st.subheader("Correlation Heatmap for Categorical and Numerical Columns")

        def point_biserial_correlation(cat_column, num_columns, data):
            # Drop rows with missing values in the columns of interest
            data = data.dropna(subset=[cat_column] + num_columns)
            categories = data[cat_column].astype('category').cat.codes
            correlations = {}
            for num_col in num_columns:
                if data[num_col].dropna().shape[0] > 0:  # Check if there are any non-NaN values
                    correlation, _ = pointbiserialr(categories, data[num_col])
                    correlations[num_col] = correlation
                else:
                    correlations[num_col] = np.nan  # Handle case with all NaN values
            return correlations

        correlations = point_biserial_correlation(selected_categorical_column, selected_numerical_columns, gender_filtered_data)

        # Create a heatmap with correlation values
        fig_heatmap_cat_num = go.Figure(data=go.Heatmap(
            z=[list(correlations.values())],
            x=list(correlations.keys()),
            y=[selected_categorical_column],
            hoverongaps=False,
            colorscale='magma',
            colorbar=dict(title='Correlation'),
        ))

        # Add text annotations with correlation values
        for i, col in enumerate(correlations.keys()):
            fig_heatmap_cat_num.add_annotation(
                x=col,
                y=selected_categorical_column,
                text=f"{correlations[col]:.2f}",
                showarrow=False,
                font=dict(size=16, color='black'),
            )

        fig_heatmap_cat_num.update_layout(
            xaxis=dict(showgrid=True, linecolor='black', tickfont=dict(color='black')),
            yaxis=dict(showgrid=True, linecolor='black', tickfont=dict(color='black')),
            hoverlabel=dict(font=dict(color='black')),
        )

        st.plotly_chart(fig_heatmap_cat_num)


    elif heatmap_selection == "Contingency table" and selected_categorical_column and selected_categorical_column_2:
        st.subheader(f"Contingency Table and Heatmap for {selected_categorical_column} vs {selected_categorical_column_2}")

        # Create contingency table
        contingency_table = pd.crosstab(filtered_data[selected_categorical_column], filtered_data[selected_categorical_column_2])

        # Calculate chi-square statistic and p-value
        chi2, p, _, _ = chi2_contingency(contingency_table)

        # Calculate Cramér's V for correlation
        cramer_v_value = cramers_v(contingency_table)

        st.write("Chi-square statistic:", chi2)
        st.write("P-value:", p)
        st.write("Cramér's V:", cramer_v_value)

        # Create heatmap for contingency table
        fig_contingency_heatmap = go.Figure(data=go.Heatmap(
            z=contingency_table.values,
            x=contingency_table.columns,
            y=contingency_table.index,
            hoverongaps=False,
            colorscale='magma',
            colorbar=dict(title='Count')
        ))

        # Add text annotations with counts and correlation values
        for i, y_value in enumerate(contingency_table.index):
            for j, x_value in enumerate(contingency_table.columns):
                count = contingency_table.iloc[i, j]
                correlation = cramer_v_value  # Use Cramér's V for correlation
                fig_contingency_heatmap.add_annotation(
                    x=x_value,
                    y=y_value,
                    text=f"{count}",  #text=f"{count}<br>Cramér's V: {correlation:.2f}"
                    showarrow=False,
                    font=dict(size=12, color='black'),
                    align='center',
                    xanchor='center',
                    yanchor='middle'
                )

        fig_contingency_heatmap.update_layout(
            xaxis=dict(title=selected_categorical_column_2, showgrid=True, linecolor='black', tickfont=dict(color='black')),
            yaxis=dict(title=selected_categorical_column, showgrid=True, linecolor='black', tickfont=dict(color='black')),
        )

        fig_contingency_heatmap.update_layout(
            xaxis_title=dict(text=selected_categorical_column_2, font=dict(color='black')),
            yaxis_title=dict(text=selected_categorical_column, font=dict(color='black')),
            hoverlabel=dict(font=dict(color='black')),
        )

        st.plotly_chart(fig_contingency_heatmap)


    else:
        st.subheader("No Heatmap Selected")



# Filter columns for plotting
non_excluded_columns = [col for col in df.columns if col not in columns_to_exclude]
container_names = [col.split('[')[0].strip() for col in non_excluded_columns]
selected_container_index = st.selectbox("Select Column for Plots", range(len(container_names)),
                                        format_func=lambda x: container_names[x])

selected_column = non_excluded_columns[selected_container_index]
selected_container_name = container_names[selected_container_index]
x_label_content = selected_column[selected_column.find('[') + 1: selected_column.find(']')] if '[' in selected_column and ']' in selected_column else selected_column

# Print total counts of filtered males, females, and intersex
gender_counts = gender_filtered_data['Gender'].value_counts()
st.subheader("Total Counts by Gender")
for gender, count in gender_counts.items():
    st.write(f"{gender}: {count}")

# Display summary for selected column
if pd.api.types.is_numeric_dtype(filtered_data[selected_column]):
    st.subheader(f"Summary for {selected_container_name}")
    if include_all_values_gender:
        st.write(f"Summary for {x_label_content} (All Genders):")
    else:
        st.write(f"Summary for {x_label_content} ({selected_gender}):")

    mean_value = gender_filtered_data[selected_column].mean()
    median_value = gender_filtered_data[selected_column].median()
    std_dev_value = gender_filtered_data[selected_column].std()
    iqr_value = np.percentile(gender_filtered_data[selected_column], 75) - np.percentile(gender_filtered_data[selected_column], 25)

    st.write(f"Minimum Value: {gender_filtered_data[selected_column].min()}")
    st.write(f"Maximum Value: {gender_filtered_data[selected_column].max()}")
    st.write(f"Mean: {round(mean_value, 2)}")
    st.write(f"Median: {round(median_value, 2)}")
    st.write(f"Standard Deviation: {round(std_dev_value, 2)}")
    st.write(f"IQR: {round(iqr_value, 2)}")

# Plot histogram and box plot if the selected column is numeric
if pd.api.types.is_numeric_dtype(filtered_data[selected_column]):
    # Histogram
    fig_hist = px.histogram(
        gender_filtered_data,
        x=selected_column,
        color='Gender',
        labels={selected_column: x_label_content, 'Count': 'Count'},
        title=f"Histogram for {selected_container_name}",
        barmode='group',
        opacity=0.7,
        color_discrete_sequence=['green', 'red', 'blue'],
    )

    fig_hist.update_layout(
        xaxis_title=dict(text=selected_column, font=dict(color='black')),
        yaxis_title=dict(text='Count', font=dict(color='black')),
    )

    fig_hist.update_traces(marker=dict(line=dict(color='black', width=1)))

    fig_hist.update_layout(
        xaxis=dict(showgrid=True, linecolor='black', tickfont=dict(color='black')),
        yaxis=dict(showgrid=True, linecolor='black', tickfont=dict(color='black')),
        hoverlabel=dict(font=dict(color='black')),
        title=dict(font=dict(size=20)),
    )

    # Box Plot
    fig_box = px.box(
        gender_filtered_data,
        x='Gender',
        y=selected_column,
        color='Gender',
        labels={selected_column: x_label_content, 'Gender': 'Gender'},
        title=f"Box Plot for {selected_container_name}",
        points='all',
        color_discrete_sequence=['green', 'red', 'blue'],
    )

    fig_box.update_layout(
        xaxis_title=dict(text=selected_column, font=dict(color='black')),
        yaxis_title=dict(text='Value', font=dict(color='black')),
        hoverlabel=dict(font=dict(color='black')),
    )

    # Add an additional box for 'All Genders'
    if include_all_values_gender:
        all_genders_data = df[df['Gender'].isin(['Male', 'Female'])]
        fig_box.add_trace(go.Box(
            y=all_genders_data[selected_column],
            name='All Genders',
            marker=dict(color='gray'),
            boxpoints='all',
        ))

    fig_box.update_layout(
        xaxis_title=dict(text=selected_column, font=dict(color='black')),
        yaxis_title=dict(text='Value', font=dict(color='black')),
    )

    fig_box.update_layout(
        xaxis=dict(showgrid=True, linecolor='black', tickfont=dict(color='black')),
        yaxis=dict(showgrid=True, linecolor='black', tickfont=dict(color='black')),
        hoverlabel=dict(font=dict(color='black')),
        title=dict(font=dict(size=20)),
    )

    # Display plots
    st.plotly_chart(fig_hist)
    st.plotly_chart(fig_box)

# Plot bar plot and pie chart if the selected column is not numeric
else:
    bar_plot_data = gender_filtered_data.groupby(['Gender', selected_column]).size().reset_index(name='Count')

    all_categories = df[selected_column].unique()
    all_gender_categories = pd.DataFrame({'Gender': df['Gender'].unique()})
    all_categories_df = pd.DataFrame({selected_column: all_categories})

    cartesian_product = pd.merge(all_gender_categories.assign(key=1), all_categories_df.assign(key=1), on='key').drop('key', axis=1)
    bar_plot_data = pd.merge(cartesian_product, bar_plot_data, how='left', on=['Gender', selected_column])

    bar_plot_data['Percentage'] = bar_plot_data.groupby('Gender')['Count'].transform(
        (lambda x: x / x.sum() * 100)).round(2)

    fig_bar = px.bar(
        bar_plot_data,
        x=selected_column,
        y='Count',
        color='Gender',
        labels={selected_column: 'Value', 'Count': 'Count'},
        title=f"Bar Plot for {selected_container_name}",
        barmode='group',
        text='Count',
        color_discrete_sequence=['green', 'red', 'blue'],
    )

    fig_bar.update_layout(
        xaxis_title=dict(text=selected_column, font=dict(color='black')),
        yaxis_title=dict(text='Count', font=dict(color='black')),
    )

    fig_bar.update_traces(textfont=dict(color='black'))

    fig_bar.update_layout(
        xaxis=dict(type='category', categoryorder='array', categoryarray=all_categories, showgrid=True, linecolor='black', tickfont=dict(color='black')),
        yaxis=dict(showgrid=True, linecolor='black', tickfont=dict(color='black')),
        hoverlabel=dict(font=dict(color='black')),
        title=dict(font=dict(size=20)),
    )

    fig_bar.update_traces(marker=dict(line=dict(color='black', width=1)), textangle=0)

    # Pie chart
    pie_chart_data = gender_filtered_data[selected_column].value_counts().reset_index(name='Count')
    fig_pie = px.pie(
        pie_chart_data,
        values='Count',
        names='index',
        title=f"Pie Chart for {selected_container_name}",
        labels={'index': selected_column, 'Count': 'Count'},
        color_discrete_sequence=px.colors.qualitative.Set3,
    )

    fig_pie.update_traces(textinfo='percent', textfont=dict(color='black'))

    fig_pie.update_layout(
        xaxis=dict(showgrid=True, linecolor='black', tickfont=dict(color='black')),
        yaxis=dict(showgrid=True, linecolor='black', tickfont=dict(color='black')),
        hoverlabel=dict(font=dict(color='black')),
        title=dict(font=dict(size=20)),
    )

    # Display plots
    st.plotly_chart(fig_bar)
    st.plotly_chart(fig_pie)