# Binarization: histogram-based gradient boosting
# This is the technique used by LightGBM and EvoTrees
# Key benefits:
# 1 - Comparing UInt8 is faster than floats during split search
# 2 - Instead of searching over all unique float values,
#     only nbins thresholds need to be evaluated

# Reference:
# https://apxml.com/courses/julia-for-machine-learning/chapter-2-julia-data-manipulation-preparation/julia-data-transformation
# https://github.com/milankl/Jenks.jl
# https://medium.com/@adnan.mazraeh1993/comprehensive-guide-to-binning-discretization-in-data-science-from-basics-to-super-advanced-006c2e215a9f

# https://github.com/sisl/Discretizers.jl
# https://github.com/myersm0/SymbolicApproximators.jl
# https://bkamins.github.io/julialang/2020/12/11/binning.html
# https://github.com/carstenbauer/BinningAnalysis.jl
# https://github.com/kirklong/BinnedStatistics.jl

"""
    sampled_quantile(X::AbstractMatrix{T}; feature_names, nbins, rng=Random.TaskLocalRNG()) where {T}
    sampled_quantile(df; feature_names, nbins, rng=Random.TaskLocalRNG())

Get the histogram breaking points of the feature data and
Transform feature data into a UInt8 sampled_quantiled matrix.
"""
function sampled_quantile(X::Matrix{T}; nbins::Int, rng::AbstractRNG) where {T<:Real}
    nrows, nfeats = size(X)
    nobs = min(nrows, 1000 * nbins)
    idx = sample(rng, 1:nrows, nobs, replace=false, ordered=true)

    # edges = Vector{Vector{T}}(undef, nfeats)
    edges = Vector{Vector{T}}(undef, nfeats)
    # featbins = Vector{UInt8}(undef, nfeats)
    # feattypes = trues(nfeats) # forse non serve, sono tutti uni
    x_bin = Matrix{UInt8}(undef, nrows, nfeats)

    Threads.@threads for j in 1:nfeats
        edges[j] = quantile(view(X, idx, j), (1:nbins-1) / nbins)
        length(edges[j]) == 1 && (edges[j] = [minimum(view(X, idx, j))])
        # featbins[j] = length(edges[j]) + 1
        x_bin[:, j] .= searchsortedfirst.(Ref(edges[j]), view(X, :, j))
    end

    # return edges, featbins, feattypes
    return edges, x_bin
end

# Binning, also known as discretization, is the process of converting continuous data into discrete intervals or bins. This technique is commonly used in data preprocessing, especially when you’re working with algorithms that prefer or require categorical data, or when you’re dealing with data that needs simplification for better interpretability. Let’s go from the basics to advanced concepts:
# Basic Concept of Binning

# In the simplest terms, binning involves grouping a set of continuous values into a smaller number of ranges, or “bins,” that summarize the data. For example, if you have a dataset of ages ranging from 18 to 65, you might group the ages into bins such as “18–30,” “31–45,” “46–60,” and “61–65.”
# Why Binning?

#     Simplicity: It simplifies complex data and helps in understanding trends.
#     Noise reduction: Helps in reducing the effect of outliers and noisy data.
#     Improves model performance: Some algorithms perform better when the input features are in discrete ranges (e.g., decision trees, some clustering algorithms).

# Types of Binning

#     Equal-width Binning:
#     In equal-width binning, the data is divided into intervals of equal size. For example, if you have 100 values between 0 and 1000, each bin might cover a range of 100 (e.g., 0–100, 101–200, etc.).
#     Equal-frequency Binning:
#     In equal-frequency binning, each bin contains the same number of data points. For instance, if you have 1000 data points, you could create 10 bins, and each bin would contain 100 points.
#     Custom Binning:
#     Custom binning allows you to define the bin edges manually based on domain knowledge or specific needs. For example, you might decide that ages should be grouped in bins of “18–30,” “31–50,” and “51–70.”
#     Clustering-Based Binning:
#     Instead of manually defining the bins, clustering techniques (like k-means) can be used to group the data based on patterns or similarities.
#     Boundary Binning:
#     Boundary binning defines boundaries that are based on specific thresholds rather than the distribution of the data.

# Steps Involved in Binning (Discretization)

#     Choose the binning strategy: Decide on one of the binning techniques, e.g., equal-width, equal-frequency, or custom.
#     Determine the number of bins: This might be based on the range of the data and the granularity of the bins you need.
#     Assign values to bins: Based on the chosen technique, you assign each value to the appropriate bin.
#     Analyze and validate the result: After binning, inspect the data to ensure that the new bins are meaningful and representative of the original data.

# Advanced Concepts of Binning

#     Supervised vs. Unsupervised Binning:

#     Unsupervised binning doesn’t use any target variable and is based solely on the data distribution.
#     Supervised binning, on the other hand, involves using the target variable to define the bins, making sure the bins maximize the relationship between the features and the target variable. Techniques like decision trees or supervised discretization methods are often used here.

#     Entropy-based Binning (Supervised):
#     This method uses information theory (entropy) to create bins that minimize the entropy of the target variable. It’s commonly used in supervised binning where the goal is to maximize predictive power.
#     Optimal Binning:
#     This involves using advanced statistical techniques to determine the “best” bin edges for your data. For example, using methods like the optimal discretization algorithm that minimizes the error between the continuous and binned representations.
#     Binning for Time Series Data:
#     When working with time series data, binning can be applied to aggregate data over time intervals. For example, you could discretize temperature data into bins representing different seasons.
#     Binning in Feature Engineering:
#     In advanced machine learning pipelines, binning is often used as a feature engineering technique. Instead of raw continuous data, models might benefit from features that represent intervals or ranges, making the relationships between features more interpretable.

# Get Adnan Mazraeh’s stories in your inbox

# Join Medium for free to get updates from this writer.

# Remember me for faster sign in
# Python Libraries for Binning

#     Pandas:
#     The pandas library provides a simple way to perform binning. The cut() and qcut() functions are commonly used for this.

#     pd.cut(): Creates bins based on equal width or custom intervals.
#     pd.qcut(): Creates bins with equal frequency.

#     Example:

#     import pandas as pd data = [1, 7, 5, 10, 9, 2, 3, 8, 4, 6] bins = pd.cut(data, bins=3) print(bins)

#     Scikit-learn:
#     Scikit-learn offers a KBinsDiscretizer which allows for more control over the binning process. It can perform equal-width, equal-frequency, and custom binning methods.
#     Example:

#     from sklearn.preprocessing import KBinsDiscretizer data = [[1], [7], [5], [10], [9], [2], [3], [8], [4], [6]] scaler = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform') binned_data = scaler.fit_transform(data) print(binned_data)

#     BinningPy:
#     BinningPy is a library specifically built for binning and discretization. It allows for advanced binning techniques.
#     FuzzyCMeans (for clustering-based binning):
#     If you want to perform clustering-based binning, you could use fuzzy clustering techniques such as fuzzy-c-means.

# R Libraries for Binning

#     Base R Functions:
#     In R, binning can be performed using base functions like cut() and quantile().

#     cut() is used to divide the data into intervals.
#     quantile() helps to divide data into quantiles, which can be used for equal-frequency binning.

#     Example:

#     data <- c(1, 7, 5, 10, 9, 2, 3, 8, 4, 6) bins <- cut(data, breaks=3) print(bins)

#     dplyr:
#     The dplyr package is often used for data manipulation in R, and its mutate() and ntile() functions can be used for binning tasks.

#     library(dplyr) data <- data.frame(values = c(1, 7, 5, 10, 9, 2, 3, 8, 4, 6)) data %>% mutate(bin = ntile(values, 3))

#     Discretization Package:
#     R has a package called discretization that provides several methods for supervised discretization, including methods based on entropy, decision trees, and others.

# Use Cases for Binning

#     Medical Data: For example, categorizing blood pressure readings into “normal,” “pre-hypertension,” and “hypertension” categories.
#     Sales Data: Categorizing revenue into “low,” “medium,” and “high” sales categories.
#     Finance: Grouping income data into different tax brackets.
#     Image Processing: Binning pixel intensity values for better image segmentation.

# Conclusion

# Binning is an essential technique in data preprocessing, offering a way to simplify and categorize continuous data. From basic equal-width or equal-frequency binning to more advanced supervised and clustering-based methods, binning can improve the interpretability and predictive power of models, especially for algorithms that perform better with categorical data. The choice of binning method depends on the nature of the data, the problem you’re solving, and the algorithm you’re using. Both Python and R provide excellent libraries to perform these tasks efficiently.