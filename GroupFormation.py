"""
Mentoring circle group formation

This code takes three input csv files and outputs mentoring circle groups as a csv file
Input 1) Mentor pairs with the following columns:
    ['Mentor Number', 'University', 'Major', 'Gender', 'Race', 'Firstgen']
Input 2) Student data subset for grouping with the following columns:
    ['Student Number', 'Graduation', 'University', 'Major', 'Gender', 'Race', 'Firstgen', 'Interest', 'Degree', 'Research']
Input 3) Student source data for the output file (Input 2 is a subset of Input 3)
    All of the columns from Input 2 and any additional columns in source data
Output) Student data from Input 3 grouped into mentoring circles numbered by mentor pair from Input 1

This code uses k-means clustering to create equal sized groups with similar interests

This code is written with cell structure for testing and debugging in Spyder

Authors: Nate Tompkins & Jackie McDermott
Created: May 2024
Updated: January 2025
"""

#%% Cell 1: Package Imports

import numpy as np
import pandas as pd
import csv
from sklearn.cluster import KMeans
from collections import Counter
from collections import deque
from datetime import datetime

#%% Cell 2: Definitions

# Assign weights applies a weight to the binary values along a named column 
# Use: df_w = assign_weights(df, df_w, 'Column', weight)
def assign_weights(df, df_w, column, weight):
    
    # For each column that was encoded create a list of each unique value
    unique_values = df[column].str.split(',').explode().unique()
    
    # For each unique value within that list assign a weight
    data_values = {f'{column}_{val}': weight for val in unique_values}
    
    # Apply this weight to the encoded data
    for column, val in data_values.items():
        if column in df_w.columns:
            df_w[column] = df_w[column] * val
            
    # Return the modifed DataFrame
    return df_w
    
# Balance clusters attempts to ensure that all mentoring circles are the same size
# Use: balanced_labels = balance_clusters(labels_groups, n_clusters, student_interests)
def balance_clusters(labels, n_clusters, interests, max_iterations=1000, recent_move_limit=10):
    
    # From the group labels count the number of people in each group and identify the target size
    cluster_counts = Counter(labels)
    target_size = len(labels) // n_clusters
    iterations = 0
    stopped_reason = ""
    
    # Keep track of recent moves to avoid moving points back to their original clusters too soon
    recent_moves = deque(maxlen=recent_move_limit)
    
    # Move students from larger groups to smaller groups
    while iterations < max_iterations:
        # Calculate centroids for each cluster
        centroids = np.array([interests[labels == i].mean(axis=0) for i in range(n_clusters)])
        
        # Check if all clusters are balanced
        if all(count <= target_size for count in cluster_counts.values()):
            stopped_reason = "All clusters are balanced."
            break
        
        moved = False
        
        # Identify the most overpopulated cluster
        overpopulated_label = max(cluster_counts, key=lambda k: cluster_counts[k])
        if cluster_counts[overpopulated_label] <= target_size:
            stopped_reason = "All clusters are balanced."
            break
        
        # Find the student furthest from the centroid in the overpopulated cluster
        indices_in_cluster = np.where(labels == overpopulated_label)[0]
        distances = np.linalg.norm(interests[indices_in_cluster] - centroids[overpopulated_label], axis=1)
        furthest_index = indices_in_cluster[np.argmax(distances)]
        
        # Find the new cluster with the centroid closest to this student
        distances_to_centroids = np.linalg.norm(centroids - interests[furthest_index], axis=1)
        
        # Exclude clusters that are already at or above target size
        for i in range(n_clusters):
            if cluster_counts[i] > target_size or (furthest_index, i) in recent_moves:
                distances_to_centroids[i] = np.inf
        
        # Select the new cluster
        new_label = np.argmin(distances_to_centroids)
        
        # If no valid new cluster found, break
        if distances_to_centroids[new_label] == np.inf:
            stopped_reason = "No additional valid moves available."
            break
        
        # Reassign the point to the new cluster
        labels[furthest_index] = new_label
        cluster_counts[overpopulated_label] -= 1
        cluster_counts[new_label] += 1
        
        # Record the move
        recent_moves.append((furthest_index, new_label))
        
        moved = True
        
        iterations += 1
        
        if not moved:
            stopped_reason = "No points were moved."
            break
    
    if iterations == max_iterations:
        stopped_reason = "Reached maximum iterations. {str(max_iterations)} moves made."

    print(f"Stopped because: {stopped_reason} {str(iterations)} moves made.")
    return labels

# Ensure that the student data and mentor data have the same columns
# Use: df_sw, df_mw = ensure_same_columns(df_sw, df_mw)
def ensure_same_columns(df1, df2, fill_value=0):
    """
    Ensure two DataFrames have the same columns by adding any missing columns.
    
    Parameters:
    df1 (pd.DataFrame): The first DataFrame.
    df2 (pd.DataFrame): The second DataFrame.
    fill_value: The value to use for filling missing columns.
    
    Returns:
    (pd.DataFrame, pd.DataFrame): The two DataFrames with the same columns.
    """
    # Get the set of columns for each DataFrame
    columns_df1 = set(df1.columns)
    columns_df2 = set(df2.columns)
    
    # Identify missing columns in each DataFrame
    missing_in_df1 = columns_df2 - columns_df1
    missing_in_df2 = columns_df1 - columns_df2
    
    # Create DataFrames for the missing columns with the fill value
    df1_missing = pd.DataFrame({col: [fill_value] * len(df1) for col in missing_in_df1})
    df2_missing = pd.DataFrame({col: [fill_value] * len(df2) for col in missing_in_df2})
    
    # Concatenate the original DataFrames with the missing columns DataFrames
    df1 = pd.concat([df1, df1_missing], axis=1)
    df2 = pd.concat([df2, df2_missing], axis=1)
    
    # Ensure the same column order in both DataFrames
    df1 = df1[sorted(df1.columns)]
    df2 = df2[sorted(df2.columns)]
    
    return df1, df2

# Averages the values for two neighboring items, creates average for each mentor pair
# Use: mentor_centroids = ave_neighbors(mentor_interests)
def ave_neighbors(lst):
    return [(lst[i] + lst[i + 1])/2 for i in range(0, len(lst), 2)]

# Returns every other value, uses the first mentor number as group identifier
# Use: mentors_names = every_other(mentors)
def every_other(lst):
    return [lst[i] for i in range(0, len(lst), 2)]

# Searches input csv file for group identifiers using student numbers
# Use: result = search_csv_column(input_csv, column_index, ids, encoding)
def search_csv_column(csv_file, column_index, identifying_numbers, encoding='utf-8'):
    result_rows = []
    with open(csv_file, 'r', encoding=encoding) as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) > column_index and str(row[column_index]) in map(str, identifying_numbers):
                result_rows.append(row)
    return result_rows

# Filters input csv into grouped output csv using group identifiers
# Use: filter_csv_by_identifiers(input_csv, output_csv, groups)
def filter_csv_by_identifiers(input_csv, output_csv, identifiers, column_index=0, encoding='utf-8'):
    with open(input_csv, mode='r', newline='', encoding='utf-8') as input_file, \
     open(output_csv, mode='w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file)
        reader = csv.reader(input_file)
        first_row = next(reader)
        writer.writerow(first_row)
        for i, ids in enumerate(identifiers, start=1):
            if i > 1:
                writer.writerow([])  # Insert a blank row for line break between subgroups
            writer.writerow(['Group {}'.format(i)])  # Insert label for the subgroup
            result = search_csv_column(input_csv, column_index, ids, encoding)
            writer.writerows(result)

#%% Cell 3: Data Import

# Load the CSV file for mentor pairs (Input 1)
df_m = pd.read_csv('SampleMentorData.csv')
# df_m is a DataFrame with raw mentor pairing data

# Load the CSV file for student grouping data (Input 2)
df_s = pd.read_csv('SampleStudentDataSubset.csv')
# df_s is a DataFrame with raw student data grouping data

# # The following is for testing/debugging importing grouping data
# # Sample mentor data:
# # Sample student data:
# # Display the first few rows to understand the structure
# print(df_m.head())
# print(df_s.head())

#%% Cell 4: Data Encoding

# List the columns to be encoded numerically, must match source data structure
# Sample Mentor Columns: MentorNumber, University, Major, Gender, Race, Firstgen
# Sample Student Columns: StudentNumber, University, Graduation, Major, Interest, Degree, Research, Race, Gender, Firstgen
# Don't include the mentor number/student number column in this list
mentor_columns = ['University', 'Major', 'Gender', 'Race', 'Firstgen']
student_columns = ['University', 'Graduation', 'Major', 'Interest', 'Degree', 'Research', 'Race', 'Gender', 'Firstgen']

# Perform one-hot encoding - split every option into its own column with binary values for yes=1, no=0
# df_b is a DataFrame with data encoded in binary
df_sb = df_s.copy()
df_mb = df_m.copy()
for col in mentor_columns:
    df_mb = pd.concat([df_mb, df_mb[col].str.get_dummies(sep=',').add_prefix(f'{col}_')], axis = 1).drop(labels=col, axis=1)
  
for col in student_columns:
    df_sb = pd.concat([df_sb, df_sb[col].str.get_dummies(sep=',').add_prefix(f'{col}_')], axis = 1).drop(labels=col, axis=1)

#%% Cell 5: Apply Weights

# Apply weights to the columns, the larger the value the more importance on that item during clustering
# Mentor weights creates the initial mentoring group location
# Student weights influences student grouping

# Assign weights to the DataFrame for each encoded column across all encodings
df_mw = df_mb.copy()
df_mw = assign_weights(df_m, df_mw, 'University', 5)
df_mw = assign_weights(df_m, df_mw, 'Major', 30)
df_mw = assign_weights(df_m, df_mw, 'Race', 5)
df_mw = assign_weights(df_m, df_mw, 'Firstgen', 5)

# Weighting Gender separetely, with Man and Woman the same but More different
df_mw['Gender_Woman'] = df_mw['Gender_Woman']*0
df_mw['Gender_Man'] = df_mw['Gender_Man']*0
# Uncomment the line below if mentors have this tag
df_mw['Gender_More'] = df_mw['Gender_More']*5

# Assign weights to the DataFrame for each encoded column across all encodings
df_sw = df_sb.copy()
df_sw = assign_weights(df_s, df_sw, 'University', 5)
df_sw = assign_weights(df_s, df_sw, 'Graduation', 50)
df_sw = assign_weights(df_s, df_sw, 'Major', 30)
df_sw = assign_weights(df_s, df_sw, 'Interest', 2)
df_sw = assign_weights(df_s, df_sw, 'Degree', 10)
df_sw = assign_weights(df_s, df_sw, 'Research', 5)
df_sw = assign_weights(df_s, df_sw, 'Race', 5)
df_sw = assign_weights(df_s, df_sw, 'Firstgen', 5)

# Weighting Gender separetely, with Man and Woman the same but More different
df_sw['Gender_Woman'] = df_sw['Gender_Woman']*0
df_sw['Gender_Man'] = df_sw['Gender_Man']*0
# Uncomment the line below if students have this tag
df_sw['Gender_More'] = df_sw['Gender_More']*5

# Ensure student and mentor DataFrames have the same columns
df_sw, df_mw = ensure_same_columns(df_sw, df_mw)

#%% Cell 6: Create Groups

# Create a list of mentoring group identifiers from the first mentor number of each pairing
mentors = df_mw['Number'].tolist()
mentors_names = every_other(mentors)

# Average weighted data from mentors to create initial group centroid
mentor_interests = df_mw.drop(['Number'],axis=1).to_numpy()
mentor_centroids = ave_neighbors(mentor_interests)

#%% Cell 7: Cluster Students

# Create a list of student identifiers from student number
students = df_sw['Number'].tolist()

# Extract every other data column as student interests
student_interests = df_sw.drop(['Number'],axis=1).to_numpy()

# Perform kmeans clustering, each student assigned a label (group number)
n_clusters = 13
kmeans_groups = KMeans(n_clusters=n_clusters, n_init=1, init=mentor_centroids, algorithm='elkan').fit(student_interests)
labels_groups = kmeans_groups.labels_

# Create groups based on initial labels
groups = [[] for _ in range(n_clusters)]
for i, label in enumerate(labels_groups):
    groups[label].append(students[i])
    
# # The following is for testing/debugging student clustering

# # Display group number label for each student (list of students with group number)
# print('Initial Labels')
# print('Initial Cluster Labels:', labels_groups)

# # Display groups with student numbers (list of groups with student number)
# print('Initial Groups')
# for idx, group in enumerate(groups):
#    print(f"Group {idx + 1}: Mentor: {mentors_names[idx]} Students: {group}")

# # Display the size of each group
# group_sizes = [len(item) for item in groups]
# print('Group Sizes:')
# print(group_sizes)

#%% Cell 8: Balance Groups

# Balance labels to similar sizes
balanced_labels = balance_clusters(labels_groups, n_clusters, student_interests)

# Create groups based on balanced labels
groups = [[] for _ in range(n_clusters)]
group_interests = [[] for _ in range(n_clusters)]
for i, label in enumerate(labels_groups):
    groups[label].append(students[i])
    group_interests[label].append(student_interests[i])

# # The following is for testing/debugging group balancing

# # Display group number label for each student (list of students with group number)
# print('Balanced Labels')
# print('Balanced Cluster Labels:', labels_groups)

# # Display groups with student numbers (list of groups with student number)
# print('Balanced Groups')
# for idx, group in enumerate(groups):
#    print(f"Group {idx + 1}: Mentor: {mentors_names[idx]} Students: {group}")
   
# # Display the size of each group
# group_sizes = [len(item) for item in groups]
# print('Group Sizes:')
# print(group_sizes)

#%% Cell 9: Output Student Group Data (use balanced groups to create Output from Input 3)

# Specify student source data (Input 3)
input_csv = 'SampleStudentData.csv' 
# Sample student data:

# Create ouput file name from current date and time
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
output_name = f"SampleStudentGroups_{formatted_datetime}.csv"

# Specify grouped student output file (Output)
output_csv = output_name 

# Create output file
filter_csv_by_identifiers(input_csv, output_csv, groups)
print("Student groups output csv file generated successfully!")
