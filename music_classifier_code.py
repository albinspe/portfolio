import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
import umap.umap_ as umap
from sklearn import manifold
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches


def read_file_create_dataframe(file_name):
    # Reads file and creates a dataframe and labels, removes missing values

    dataframe = pd.read_csv(file_name, sep=",")

    # Removes labels from data
    dataframe = dataframe.drop(columns=["spectral_decrease"])

    try:
        dataframe = dataframe.drop(columns=["title"])
    except KeyError:
        pass

    print(dataframe.isna().sum())
    print("Antal rader med minst en NaN:", dataframe.isna().any(axis=1).sum())

    dataframe = dataframe.dropna()

    genres = dataframe["genre"]

    dataframe = dataframe.drop(columns=["genre"])

    print("Antal låtar kvar:", dataframe["bpm"].shape)

    scaler = StandardScaler()
    dataframe = pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns)

    return dataframe, genres


def calculate_silhoutte_score(dataframe, k):
    # Calculates silhouette score for k-means clustering

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(dataframe)

    labels = kmeans.predict(dataframe)
    centroids = kmeans.cluster_centers_

    silhouette_labels = metrics.silhouette_samples(dataframe, labels)
    silhouette_score = silhouette_labels.mean()

    return silhouette_score


def k_means_clustering(dataframe):
    # Finds best k-value from silhouette score and performs k-means clustering

  silhouette_scores = {}

  for k in range(2, 7):
    silhouette_score = calculate_silhoutte_score(dataframe, k)
    silhouette_scores[k] = silhouette_score

  best_k = max(silhouette_scores, key=silhouette_scores.get)

  print("Best k for k-means clustering", best_k)

  kmeans = KMeans(n_clusters=best_k, random_state=42)
  kmeans.fit(dataframe)

  labels = kmeans.predict(dataframe)
  centroids = kmeans.cluster_centers_

  return labels, centroids, best_k


def plot_eigenvectors(dataframe):
    # Plots eigenvalues, eigenvectors and finds number of componenets that remain 95% of variance

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(dataframe)
    data_scaled = pd.DataFrame(scaled_values, columns=dataframe.columns)

    covariance = data_scaled.cov()
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    # Sorts indices
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Plots eigenvalues
    plt.figure()
    plt.plot(eigenvalues, "ro-")
    plt.title("Eigenvalues")
    plt.tight_layout()
    plt.show()

    Q_PC1_PC2 = eigenvectors[:, :2]
    n_components = min(15, eigenvectors.shape[1])
    Q_PC1_PC15 = eigenvectors[:, :n_components]

    vec1 = Q_PC1_PC2[:,0]
    vec2 = Q_PC1_PC2[:,1]

    plt.figure(figsize=(10,5))
    plt.plot(np.abs(vec1), 'ro--')
    plt.plot(np.abs(vec2), 'bo--')
    plt.xticks(ticks=range(len(data_scaled.columns)), labels=data_scaled.columns, rotation=90)
    plt.title("Absolute values of eigenvectors with the highest eigenvalues")
    plt.tight_layout()
    plt.grid()

    plt.show()

    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--', label="95% variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance by Number of Components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return Q_PC1_PC2, Q_PC1_PC15


def pca(dataframe, Q1, Q2, predicted_labels, true_labels, k_means_centroids):
    # Performs PCA and visualize data in two dimensions

    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(dataframe)
    data_scaled = pd.DataFrame(scaled_values, columns=dataframe.columns)

    x = data_scaled.values

    mu = np.mean(x, axis=0)
    x_mean_centered = x - mu

    x_proj_2d = np.dot(x_mean_centered, Q1)
    x_proj_15d = np.dot(x_mean_centered, Q2)
    k_means_centroids_proj = np.dot(k_means_centroids - mu, Q1)

    plt.figure(figsize=(7,7))
    plt.scatter(x_proj_2d[:, 0], x_proj_2d[:, 1], c=predicted_labels, s=5)
    plt.scatter(k_means_centroids_proj[:, 0], k_means_centroids_proj[:, 1], marker="*", c="black", s=100)
    plt.title("PCA dimensions 1 and 2 (k-means clusters centroids marked)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.xlim(-20, 40)
    plt.ylim(-50, 25)
    plt.tight_layout()
    plt.show()

    genre_to_color = {"Jazz":"b", "Electronic":"y", "Rock":"g", "Pop":"r", "Classical":"m", "Hip-Hop":"brown"}

    color_vector = np.array([genre_to_color[label] for label in true_labels])

    legend_patches = [mpatches.Patch(color=color, label=genre) for genre, color in genre_to_color.items()]

    plt.figure(figsize=(7, 7))
    plt.scatter(x_proj_2d[:, 0], x_proj_2d[:, 1], c=color_vector, s=5)
    plt.title("PCA dimensions 1 and 2 (True genres marked)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(handles=legend_patches, title="Genre", loc="upper right")
    plt.xlim(-20,40)
    plt.ylim(-50,25)
    plt.tight_layout()
    plt.show()

    pca_dataframe_15d = pd.DataFrame(x_proj_15d, columns=["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "PC11", "PC12", "PC13", "PC14", "PC15"])

    return pca_dataframe_15d


def lda(dataframe, labels):

    X = dataframe
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_scaled, y_train)

    y_pred_lda = lda.predict(X_test_scaled)

    print("Accuracy:", accuracy_score(y_test, y_pred_lda))
    print("\nClassification report:\n", classification_report(y_test, y_pred_lda))

    # Confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred_lda, labels=lda.classes_)

    # Normalize to percentage
    conf_mat_percent = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis] * 100

    # Plot
    sns.heatmap(conf_mat_percent, annot=True, fmt=".1f", cmap="Blues", xticklabels=lda.classes_,
                yticklabels=lda.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix for LDA (in %)")
    plt.tight_layout()
    plt.show()


def qda(dataframe, labels):

    X = dataframe
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train_scaled, y_train)

    y_pred_qda = qda.predict(X_test_scaled)

    print("Accuracy:", accuracy_score(y_test, y_pred_qda))
    print("\nClassification report:\n", classification_report(y_test, y_pred_qda))

    # Confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred_qda, labels=qda.classes_)

    # Normalize to percentage
    conf_mat_percent = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis] * 100

    # Plot
    sns.heatmap(conf_mat_percent, annot=True, fmt=".1f", cmap="Blues", xticklabels=qda.classes_,
                yticklabels=qda.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix for QDA (in %)")
    plt.tight_layout()
    plt.show()


def find_best_trees_rf(dataframe, labels):
    # Finds best number of decision trees for random forest

    X = dataframe
    y = labels

    X_train_rf, X_test_val, y_train_rf, y_test_val = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42, stratify=y_test_val)

    scaler = StandardScaler()
    X_train_rf_scaled = scaler.fit_transform(X_train_rf)
    X_val_rf_scaled = scaler.transform(X_val)
    X_test_rf_scaled = scaler.transform(X_test)

    n_classifiers_list = [100, 200, 300, 400, 500, 600]
    accuracy_list = []

    for n in n_classifiers_list:

      rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
      rf.fit(X_train_rf_scaled, y_train_rf)

      y_pred_rf = rf.predict(X_val_rf_scaled)

      accuracy_list.append(accuracy_score(y_val, y_pred_rf))

    plt.plot(n_classifiers_list, accuracy_list)
    plt.xlabel("Number of estimators")
    plt.ylabel("Accuracy")
    plt.title("Accuracy for differnt amounts of decision trees")
    plt.tight_layout()
    plt.show()

    max_acc = max(accuracy_list)
    max_acc_index = accuracy_list.index(max_acc)
    best_n = n_classifiers_list[max_acc_index]
    print("Number of trees with highest accuracy", n_classifiers_list[max_acc_index])

    return best_n


def rf(dataframe, labels, n_trees, five_songs=0, five_genres=0, test_five_songs=False):

    X = dataframe
    y = labels

    X_train_rf, X_test_val, y_train_rf, y_test_val = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

    X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42,
                                                    stratify=y_test_val)

    scaler = StandardScaler()
    X_train_rf_scaled = scaler.fit_transform(X_train_rf)
    X_val_rf_scaled = scaler.transform(X_val)
    X_test_rf_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=n_trees, random_state=42, n_jobs=-1)
    rf.fit(X_train_rf_scaled, y_train_rf)

    y_pred_rf = rf.predict(X_test_rf_scaled)

    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("\nClassification report:\n", classification_report(y_test, y_pred_rf))

    # Confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred_rf, labels=rf.classes_)

    # Normalize to percentage
    conf_mat_percent = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis] * 100

    # Plot
    sns.heatmap(conf_mat_percent, annot=True, fmt=".1f", cmap="Blues", xticklabels=rf.classes_,
                yticklabels=rf.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix for Random Forest (in %)")
    plt.tight_layout()
    plt.show()

    if test_five_songs is True:
        y_pred_five_songs = rf.predict(five_songs)

        # Confusion matrix
        conf_mat = confusion_matrix(five_genres, y_pred_five_songs, labels=rf.classes_)
        sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=rf.classes_, yticklabels=rf.classes_)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix for Five Songs")
        plt.tight_layout()
        plt.show()


def run_umap(dataframe, labels):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataframe)

    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)


    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels, palette="tab10", alpha=0.5)
    plt.title("UMAP projection of music features")
    plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.show()


def t_SNE(dataframe, labels):


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dataframe)

    embedding = manifold.TSNE(2,perplexity=80, random_state=42)

    X_prime = embedding.fit_transform(X_scaled)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_prime[:, 0], y=X_prime[:, 1], hue=labels, palette="tab10", alpha=0.5)
    plt.title("t-SNE projection of music features")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()


def main():

    # Reads file and creates dataframe
    raw_data, genres = read_file_create_dataframe("rensad_data_final.txt")
    five_songs, five_genres = read_file_create_dataframe("fem_låtar.txt")

    # Finds best k for k-means clustering and applies it to data
    predicted_labels, k_means_centroids, k = k_means_clustering(raw_data)

    # Plots eigenvectors with the two highest eigenvalues
    Q_2D, Q_15D = plot_eigenvectors(raw_data)

    # Performs pca and returns projected data on pc1-pc15
    pca_data = pca(raw_data, Q_2D, Q_15D, predicted_labels, genres, k_means_centroids)

    # Performs lda and qda on data and plots results
    lda(raw_data, genres)
    qda(raw_data, genres)

    # Finds best amount of trees n for random forest and performs random forest and plots results
    n = find_best_trees_rf(raw_data, genres)
    rf(raw_data, genres, n, five_songs, five_genres, test_five_songs=True)

    # Performs lda and qda on data in pc1 and pc2 dimensions
    lda(pca_data, genres)
    qda(pca_data, genres)

    # Finds best amount of trees n for random forest and performs random forest on pc1 and pc2
    n_post_pca = find_best_trees_rf(pca_data, genres)
    rf(pca_data, genres, n_post_pca)

    # Performs umap and plots results
    run_umap(raw_data, genres)

    # Performs t-SNE and plots results
    t_SNE(raw_data, genres)

main()