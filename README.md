“AI- Powered Enhanced EHR imaging & Documentation system”

     This particular project aims to enhance the EHR(Electronic Health Records) by integrating Generative Capabilities for medical image analysis and administrative automation. GenAI will be used in this project to improve and enhance the clarity & interpretability of medical imaging like X-rays, MRIs , CT Scans etc and to automate clinical documentation and ICD 10 coding. These enhancement will help to reduce the time that clinicians spend on non-clinical tasks and it will also support faster and much more accurate decision making.By the end of this project we will get some expected outcomes. Those are as follows:- 
1. There will be improved interpretation of medical images through AI-driven enhancement and reconstruction. 
2. There will be significant reduction in time, spent on documentation , through automated clinical note generation. 
3. There will be streamlined ICD 10 coding integrated into clinical workflows.
4. Most importantly,  there will be greater focus on patient care by minimising repetitive administrative efforts.
    
So here we have divided the project into 4 modules. Those are:-
1. Module 1 :- Data collection and preprocessing. 
2. Module 2 :- Medical imaging enhancement. 
3. Module 3 :- Clinical note generation and ICD 10 coding automation. 
4. Module 4 :- Integration and deployment.

(*)Module 1:-

So, the main objective of module 1 is "to prepare imaging and clinical data for AI model training and application".There are several tasks which i have done in module 1.Those are :- 
1. Collection of medical imaging datasets.(Eg. X-ray, MRI, CT SCAN, ultrasound etc.)
2. Gathering structured and unstructured EHR content including patient notes and coding data. 
3. Cleaning, labelling and standardising the data for GenAI model compatibility.

Firstly I have collected the datasets for "ICD 10 coding" , "Lab reports" , "patient records", "doctor's prescription" & "medical images" from "kaggle" . Now for the coding part I have used Google Collab. Now I have imported essential Python libraries for data manipulation (pandas, numpy) and visualization (matplotlib.pyplot, seaborn). After that I have uploaded a file from my local computer to the Google Colab environment. I have used , files.upload() prompts to select a file, and then the code extracts the contents of the uploaded zip file into a new directory named . After extracting the zip file, the cell reads the file, which is located inside the directory, into a pandas DataFrame. This DataFrame will be used for subsequent analysis. Now I have used print(df.info()) command to get a summary of the dataframe. It shows :
1.	The number of entries (rows).
2.	Each column's name.
3.	The number of non-null values for each column (helpful for identifying missing data).
4.	The data type of each column (e.g., int64, float64, object).
5.  Memory usage. This is crucial for an initial understanding of the dataset's structure and data quality. Now I have generated descriptive statistics for all numerical columns in df. For each numerical column, it outputs:-
1.	count: The number of non-null values.
2.	mean: The average value.
3.	std: The standard deviation.
4.	min: The minimum value.
5.	25%, 50% (median), 75%: The quartiles.
6.	max: The maximum value. 

These statistics help in understanding the distribution, spread, and potential outliers in my numerical data. After that I have calculated and displayed the total number of missing values for each column in the df DataFrame. It's a quick way to see which columns have missing data and how much, which guides decisions on how to handle them (e.g., imputation, dropping rows/columns). After doing it I have done visualisation through correaltion heat map. I have generated a correlation heatmap for the numerical features in the df DataFrame. First, it selects only the numerical columns. Then, numerical_df.corr() calculates the pairwise correlation between these columns. seaborn.heatmap() visualizes this correlation matrix, where annot=True displays the correlation values on the map, cmap='coolwarm' sets the color scheme, and fmt=".2f" formats the annotations to two decimal places. This plot helps in identifying relationships and dependencies between numerical variables. 

After that I have done histoplot visualization. I have created a histogram to show the distribution of the 'Age' column in df. “seaborn.histplot()” is used for this purpose, with bins=5 defining the number of bins for the histogram. The plot helps to understand the age demographics in the dataset.

After that I have done scatter plot visualisation of Age vs nWBV. Here I have generated a scatter plot to visualize the relationship between 'Age' and 'nWBV' (Normalized Whole Brain Volume) from the df DataFrame. Scatter plots are useful for observing patterns, trends, or clusters between two numerical variables. The title and axis labels provide context for the visualization.

After that I have created a pie chart to display the distribution of genders ('M/F' column) in the df DataFrame.  df['M/F'].value_counts() counts the occurrences of each gender.  plt.pie() then creates the pie chart, with labels taken from the gender categories, startangle=90 to rotate the starting point, and custom colors.  plt.axis('equal') ensures the pie chart is drawn as a circle. Now I have done the process of standardizing the dataset. It first identifies numerical and non-numerical columns in df. It then separates the df into two new DataFrames: df_numerical containing only the numerical columns and df_non_numerical containing the non-numerical ones. This separation is a common step before applying scaling techniques like StandardScaler, which are typically only applied to numerical features. The heads of both new DataFrames are displayed to show the separation.
After doing all these now I have uploaded the datasets for patient details and lab reports. Then I have extracted the data from the datasets. And then I have cleaned, labelled and standardised the data sets for model compatibility.

(*)Module 2:-

So, the main objective of module 2 is "to Use GenAI to enhance image quality and support diagnosis".There are several tasks which I have done in module 2. Those are:- 
1. Applying GenAI for denoising and realistic reconstruction of medical images. 
2. Support better visualization for Clinicians by improving image resolution and clarity. 
3. Training image enhancement modules using openAI tools.

After  uploaded a zip file, in my first step I have extracted the contents so that i can access the images. So  I created the extraction directory and then i had unzipped the file.  Now as the dataset has been extracted, I explored the directory structure of the dataset to see how the images are organized. After that I counted the number of images in each category for class balance. After that I had defined the base directory, categories and I had created a dictionary to store the counts. Then I listed all files in one directory. After doing that, I had applied GenAI and for that I had seted the environment for GenAI by installing the required libraries. Then I had loaded the data and preprocessed it for model training. Now I had set up the environment, loaded and preprocessed a subset of images. Now in my next step i had chosen suitable GenAI model for image denoising and for that I had used GANs(Generative Adversarial Networks). Now I had defined the discriminator model. Because this model will try to distinguish between the real/clean images and fake/denoised images generated by the generator. In my next step, I had defined the loss function and the optimizers for the GANs. Then I wrote the code for better clarity and better visualization of the images and then I had done a visual comparison between original and enhanced images.  Then  I had written the code to calculate PSNR (Peak Signal-to-Noised Ratio) for the images before and after enhancement. As we know, PSNR is used to measure the quality of a reconstructed image in comparison to original one. Next I had built the image classification model. Then After the classification model has been trained successfully, in my very next step I had evaluated it's performance on the test set to see how well it generalizes to unseen data.  Now I visualized some predictions along with their true and predicted labels to see how well the model is performing on individual images. 

(*)Module 3:-

So the main objective of module 3 is to Automate routine documentation and coding tasks using GenAI.There are several tasks which I have done in module 3. Those are:- 
1. Generating clinical notes from structured data and doctor observations.
2. Automating ICD-10 coding by mapping EHR input to standard classifications.
3. Decreasing documentation workload through seamless integration of GenAI tools.

Module-3 is all about "Clinical Note Generation", "automate routine documentation" and "coding tasks using GenAI" . I had uploaded the 3 different datasets i.e clinical data , patient records and lab reports for merging and further operations . After uploaded the datasets , I wrote code to merge them and display the first few rows of the merged dataset. Now after merged the dataset, I had generated a new code cell that will take a patient I'd and input and then display all the information available for that patient from the merged dataframe. Now after that, I had accessed a pre trained model from "hugging face" using the transformers library and pipeline function. Then I had summarised the patient details using the loaded model. Then I had created a list to store the text representation of each patient's details. So now I had successfully decreased the documentation workload through seamless integration of GenAI tools.

(*)Module 4:-

SO the main objective of module 4 is "to deploy and integrate the enhanced EHR features into clinical environments".There are several tasks which I have done in module 4.Those are:-
1. Deploying trained GenAI models into real-time clinical workflows.
2. Integrating with hospital EHR systems for image processing and note generation.
3. Providing onboarding sessions for medical staff to use the new tools effectively.

Module-4 is all about “deployment”. In this module I have established a FastAPI backend service designed to manage, analyze, and apply machine learning models to three specific healthcare-related datasets: heart_attack (from an Excel file), heart_failure, and lab_reports (both from CSV/Excel). The service is configured with CORS middleware to allow requests from any origin. The core functionality revolves around the load_dataset function, which efficiently loads the data into memory using a global cache (_dfs) via Pandas, handling both Excel and CSV formats. The API provides a rich set of endpoints for data exploration, including listing available datasets, previewing rows, getting column names, retrieving statistical descriptions, and fetching individual rows by index. Beyond simple retrieval, the service enables data manipulation via a /filter endpoint, which allows users to filter datasets based on various comparison operators (e.g., equals, less than, contains), and a /merge endpoint, which performs joins (like inner or outer) between any two datasets on specified keys, saving the result temporarily in memory. Crucially, the application also integrates a complete machine learning pipeline, featuring a /train endpoint that uses a dataset and a target column to train a Logistic Regression model, saving the trained model and its features to a persistent directory (/mnt/data/models) and returning its accuracy. A corresponding /predict endpoint handles real-time inference by loading a saved model, pre-processing new input data to match the training features, and returning the prediction. Finally, utility endpoints are included for listing saved models, exporting any dataset to a CSV file, and a basic health check.

















