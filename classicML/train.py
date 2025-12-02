import metrics
import visual_pipeline as vp
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    image_paths =  []
    labels = []
    source_dir = 'data/raw'
    categories = [d for d in os.listdir(source_dir)
                if os.path.isdir(os.path.join(source_dir, d))]
    for category in categories:
        category_path = os.path.join(source_dir, category)
        files = [os.path.join(category_path, f) for f in os.listdir(category_path)
                if os.path.isfile(os.path.join(category_path, f))]
        image_paths.extend(files)
        labels.extend([category] * len(files))

    print(f"Found {len(image_paths)} images in {len(categories)} categories")
    os.makedirs("metrics", exist_ok=True)

    le = LabelEncoder()
    # It is easier for metrics and evaluation to have encoded labels (numeric instead of strings)
    labels_encoded = le.fit_transform(labels)

    sift = vp.DescriptorExtractor(descriptor_type='SIFT')
    output_dir = "descriptors"
    os.makedirs(output_dir, exist_ok=True)
    for x in (image_paths):
        p = Path(x)
        output_path = os.path.join(output_dir, f"{p.stem}.npy")

        # Skip if the descriptor file already exists
        if os.path.exists(output_path):
            print(f"Skipping {x} — descriptors already saved at {output_path}")
            continue
        descriptors = sift.extract_from_file(x)

        np.save(output_path, descriptors)

    # To test different algorithms we saved the descriptors to only extract them once
    image_paths = [f"{output_dir}/{Path(x).stem}.npy" for x in image_paths]

    # Note that stratify makes sure that the class distribution is maintained in train and test splits (important in classification tasks)
    X_train, X_test, y_train, y_test = train_test_split(image_paths, labels_encoded, test_size=0.2, random_state=0, stratify=labels)

    clustering_bovw = vp.MiniBatchKMeansClustering(
        n_clusters=512,
        batch_size=10000
    )

    clustering_bovw.fit_iterative(
        data_loader=X_train,
        load_func=sift.load_descriptors
    )

    # With this approach we highly reduce training time since we don't need to constantly fit the clustering in every fold
    bovw = vp.create_bovw_pipeline(
        clustering=clustering_bovw,
        descriptor_extractor=sift.load_descriptors
    )

    from sklearn.linear_model import LogisticRegression

    bovw_softmax_pipeline = Pipeline([
            ('encoding', bovw),
            ('classifier', LogisticRegression(
                solver='lbfgs',
                max_iter=150,
                C=7.0
                # note that multi_class='multinomial' is default in recent sklearn versions when n_classes > 3
            ))
    ])
    models_dir = "models"
    results_bovw_softmax = metrics.evaluate_model(
            bovw_softmax_pipeline,
            model_name="BoVW + Softmax",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            le=le,
            verbose=True,
            save_model_path=models_dir + "/bovw_softmax.pkl",
            save_func=sift.extract_from_file
            )

    metrics.plot_confusion_matrix(results_bovw_softmax, save_path="metrics/bovw_softmax_confusion_matrix.png")
    metrics.plot_roc_curve(results_bovw_softmax, save_path="metrics/bovw_softmax_roc_curve.png")
    metrics.print_classification_report(results_bovw_softmax)
