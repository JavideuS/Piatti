import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import normalize
from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum
import cv2 as cv


class DescriptorType(Enum):
    """Enum for available descriptor types."""
    SIFT = "SIFT"
    ORB = "ORB"
    AKAZE = "AKAZE"


class DescriptorExtractor:
    """
    Unified interface for different feature descriptors.

    This class handles the quirks of different OpenCV detectors
    and provides a consistent interface.

    Parameters
    ----------
    descriptor_type : str or DescriptorType
        Type of descriptor to use ('SIFT', 'ORB', 'AKAZE', etc.)
    n_features : int, optional
        Maximum number of features to detect (for SIFT, ORB)
    **kwargs : dict
        Additional parameters for the specific detector
    """

    def __init__(
        self,
        descriptor_type: str = 'SIFT',
        n_features: Optional[int] = None,
        **kwargs
    ):
        self.descriptor_type = descriptor_type.upper()
        self.n_features = n_features
        self.kwargs = kwargs
        self.detector = self._create_detector()
        self.descriptor_size = self._get_descriptor_size()

    def _create_detector(self):
        """Create the appropriate OpenCV detector."""
        dt = self.descriptor_type

        if dt == 'SIFT':
            if self.n_features is not None:
                return cv.SIFT_create(nfeatures=self.n_features, **self.kwargs)
            return cv.SIFT_create(**self.kwargs)

        elif dt == 'ORB':
            if self.n_features is not None:
                return cv.ORB_create(nfeatures=self.n_features, **self.kwargs)
            return cv.ORB_create(**self.kwargs)

        elif dt == 'AKAZE':
            return cv.AKAZE_create(**self.kwargs)

        else:
            raise ValueError(
                f"Unknown descriptor type: {dt}. "
                f"Available: {[e.value for e in DescriptorType]}"
            )

    def _get_descriptor_size(self) -> int:
        """Get the descriptor dimensionality for each type."""
        size_map = {
            'SIFT': 128,
            'ORB': 32,
            'AKAZE': 61
        }
        return size_map.get(self.descriptor_type, 128)

    def extract(
        self,
        image: np.ndarray,
        return_keypoints: bool = False
    ) -> np.ndarray:
        """
        Extract descriptors from an image array.

        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale or color)
        return_keypoints : bool, default=False
            If True, return (keypoints, descriptors) tuple

        Returns
        -------
        descriptors : np.ndarray
            Descriptor array of shape (n_keypoints, descriptor_size)
            Returns empty array if no keypoints detected
        """
        if len(image.shape) == 3:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        keypoints, descriptors = self.detector.detectAndCompute(image, None)

        if descriptors is None or len(keypoints) == 0:
            descriptors = np.zeros((0, self.descriptor_size), dtype=np.float32)
            keypoints = []
        else:
            descriptors = descriptors.astype(np.float32)

        if return_keypoints:
            return keypoints, descriptors
        return descriptors

    def extract_from_file(
        self,
        filepath: str,
        return_keypoints: bool = False
    ) -> np.ndarray:
        """
        Extract descriptors from an image file.

        Parameters
        ----------
        filepath : str
            Path to image file
        return_keypoints : bool, default=False
            If True, return (keypoints, descriptors) tuple

        Returns
        -------
        descriptors : np.ndarray
            Descriptor array or empty array if file can't be read
        """
        image = cv.imread(str(filepath), cv.IMREAD_GRAYSCALE)

        if image is None:
            empty_desc = np.zeros((0, self.descriptor_size), dtype=np.float32)
            if return_keypoints:
                return [], empty_desc
            return empty_desc

        return self.extract(image, return_keypoints=return_keypoints)

    def extract_batch(
        self,
        images: list,
        from_files: bool = False
    ) -> list:
        """
        Extract descriptors from multiple images.

        Parameters
        ----------
        images : list
            List of image arrays or file paths
        from_files : bool, default=False
            If True, treat images as file paths

        Returns
        -------
        list of np.ndarray
            List of descriptor arrays
        """
        if from_files:
            return [self.extract_from_file(img) for img in images]
        else:
            return [self.extract(img) for img in images]

    def load_descriptors(self, filepath: str):
        """Load descriptors from a .npy file."""
        descriptors = np.load(filepath, allow_pickle=True)
        if descriptors is None or descriptors.size == 0:
            return None

        descriptors = np.asarray(descriptors, dtype=np.float32)

        if descriptors.ndim != 2 or descriptors.shape[1] != 128:
            return None

        return descriptors

    def __repr__(self):
        return (
            f"DescriptorExtractor(type={self.descriptor_type}, "
            f"n_features={self.n_features}, "
            f"descriptor_size={self.descriptor_size})"
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('detector', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            if hasattr(self, 'descriptor_type') and self.descriptor_type in ('SIFT', 'ORB', 'AKAZE'):
                self.detector = self._create_detector()
            else:
                self.detector = None
        except Exception:
            self.detector = None


class ClusteringAlgorithm(ABC):
    """Base class for clustering algorithms."""

    def __init__(self, n_clusters: int, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._model = None
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray):
        pass

    @abstractmethod
    def fit_iterative(self, data_loader, load_func: Optional[callable] = None):
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        return self.model.predict(X)

    @property
    def cluster_centers_(self) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model must be fitted first.")
        return self.model.cluster_centers_

    def descriptor_batch_generator(self,
    data_loader, load_func=None, max_descriptors=10000):
        """
        Yields batches of descriptors with a total number of rows up to max_descriptors.

        Args:
            data_loader: iterable of items to load descriptors from.
            load_func: optional function(item) -> descriptor array
            max_descriptors: max total rows per batch

        Yields:
            np.ndarray of shape (~max_descriptors, n_features)
        """
        batch = []
        current_size = 0

        for item in data_loader:
            data = load_func(item) if load_func else item

            if data is None or data.size == 0:
                continue

            if data.ndim == 1:
                data = data.reshape(1, -1)

            batch.append(data)
            current_size += data.shape[0]

            if current_size >= max_descriptors:
                yield np.vstack(batch)
                batch = []
                current_size = 0

        if batch:
            yield np.vstack(batch)

    def get_params(self, deep=True):
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    def to_dict(self):
        """Return a dictionary representation of the clustering configuration."""
        return {
            'type': self.__class__.__name__,
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'is_fitted': self._is_fitted
        }

    def __repr__(self):
        params = ', '.join(f"{k}={v!r}" for k, v in self.to_dict().items() if k != 'type')
        return f"{self.__class__.__name__}({params})"


class MiniBatchKMeansClustering(ClusteringAlgorithm):
    """MiniBatch KMeans with support for iterative fitting."""

    def __init__(self, n_clusters: int, batch_size: int = 1024, random_state: int = 42, **kwargs):
        super().__init__(n_clusters, random_state)
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.model = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=random_state,
            **kwargs
        )

    def fit(self, X: np.ndarray):
        self.model.fit(X)
        self._is_fitted = True
        return self

    def fit_iterative(
        self,
        data_loader,
        load_func: Optional[callable] = None,
        accumulate_batch_size: Optional[int] = None
    ):
        """
        Fit iteratively using partial_fit.

        Args:
            data_loader: Iterable of file paths or data chunks
            load_func: Function to load data from paths
            accumulate_batch_size: Size to accumulate before partial_fit
                                   (defaults to self.batch_size)
        """
        if accumulate_batch_size is None:
            accumulate_batch_size = self.batch_size

        for batch in self.descriptor_batch_generator(data_loader, load_func, accumulate_batch_size):
            self.model.partial_fit(batch)

        self._is_fitted = True
        return self

    def to_dict(self):
        """Return a dictionary representation including batch_size."""
        base_dict = super().to_dict()
        base_dict['batch_size'] = self.batch_size
        return base_dict


class KMeansClustering(ClusteringAlgorithm):
    """Standard KMeans clustering."""

    def __init__(self, n_clusters: int, random_state: int = 0, **kwargs):
        super().__init__(n_clusters, random_state)
        self.kwargs = kwargs
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            **kwargs
        )

    def fit(self, X: np.ndarray):
        """Fit KMeans on all data at once."""
        self.model.fit(X)
        self._is_fitted = True
        return self

    def fit_iterative(self, data_loader, load_func: Optional[callable] = None):
        """KMeans doesn't support iterative fitting - loads all data first."""
        print("Warning: KMeans doesn't support true iterative fitting. Loading all data...")

        all_data = []
        for item in data_loader:
            data = load_func(item) if load_func else item
            if data is not None and data.size > 0:
                all_data.append(data)

        if all_data:
            X = np.vstack(all_data)
            self.fit(X)
        return self


class EncodingAlgorithm(ABC):
    """Base class for encoding algorithms."""

    def __init__(self, clustering: ClusteringAlgorithm):
        self.clustering = clustering

    @abstractmethod
    def encode(self, descriptors: np.ndarray) -> np.ndarray:
        pass

    def to_dict(self):
        """Return a dictionary representation including clustering info."""
        return {
            'type': self.__class__.__name__,
            'clustering': self.clustering.to_dict()
        }

    def __repr__(self):
        return f"{self.__class__.__name__}(clustering={self.clustering!r})"


class BagOfVisualWords(EncodingAlgorithm):
    """Bag of Visual Words encoding."""

    def __init__(self, clustering: ClusteringAlgorithm,
                 normalize_hist: bool = True,
                 norm_type: str = 'l2'):
        super().__init__(clustering)
        self.normalize_hist = normalize_hist
        self.norm_type = norm_type

    def encode(self, descriptors: np.ndarray) -> np.ndarray:
        if descriptors is None or descriptors.size == 0:
            return np.zeros(self.clustering.n_clusters)

        labels = self.clustering.predict(descriptors)
        histogram = np.bincount(labels, minlength=self.clustering.n_clusters).astype(np.float32)

        if self.normalize_hist:
            if self.norm_type == 'l2':
                norm = np.linalg.norm(histogram)
                if norm > 0:
                    histogram = histogram / norm
            elif self.norm_type == 'l1':
                if histogram.sum() > 0:
                    histogram = histogram / histogram.sum()

        return histogram

    def to_dict(self):
        """Return a dictionary representation including normalization info."""
        base_dict = super().to_dict()
        base_dict['normalize_hist'] = self.normalize_hist
        base_dict['norm_type'] = self.norm_type
        return base_dict

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"normalize={self.normalize_hist}, "
                f"norm_type={self.norm_type!r}, "
                f"clustering={self.clustering!r})")


class VLAD(EncodingAlgorithm):
    """Vector of Locally Aggregated Descriptors."""

    def __init__(self, clustering: ClusteringAlgorithm, normalize_vlad: bool = True):
        super().__init__(clustering)
        self.normalize_vlad = normalize_vlad

    def encode(self, descriptors: np.ndarray) -> np.ndarray:
        if descriptors is None or descriptors.size == 0:
            n_features = self.clustering.cluster_centers_.shape[1]
            return np.zeros(self.clustering.n_clusters * n_features)

        labels = self.clustering.predict(descriptors)
        centers = self.clustering.cluster_centers_
        n_clusters, n_features = centers.shape
        vlad = np.zeros((n_clusters, n_features))

        for idx in range(n_clusters):
            mask = (labels == idx)
            if np.any(mask):
                residuals = descriptors[mask] - centers[idx]
                vlad[idx] = residuals.sum(axis=0)

        vlad = vlad.flatten()
        vlad = vlad.reshape(n_clusters, n_features)
        vlad = normalize(vlad, norm='l2', axis=1)
        vlad = vlad.flatten()
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))

        if self.normalize_vlad:
            vlad = normalize(vlad.reshape(1, -1), norm='l2').flatten()

        return vlad

    def to_dict(self):
        """Return a dictionary representation including normalization info."""
        base_dict = super().to_dict()
        base_dict['normalize_vlad'] = self.normalize_vlad
        return base_dict

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"normalize={self.normalize_vlad}, "
                f"clustering={self.clustering!r})")


class VisualEncodingTransformer(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible wrapper for visual encoding pipelines.

    This allows you to:
    - Save/load with joblib
    - Use in sklearn Pipelines
    - Use with GridSearchCV
    - Keep your flexible architecture

    Parameters
    ----------
    encoding : EncodingAlgorithm
        Custom encoding algorithm instance
    descriptor_extractor : callable, optional
        Function to extract descriptors from images
        Signature: descriptor_extractor(image_path) -> np.ndarray
    iterative_fit : bool, default=False
        If True, use iterative fitting (for large datasets)
    skip_clustering_fit: bool, default=False
        If True, skip fitting the clustering model
    """

    def __init__(
        self,
        encoding: EncodingAlgorithm,
        descriptor_extractor: Optional[callable] = None,
        iterative_fit: bool = False,
        skip_clustering_fit: bool = False
    ):
        self.encoding = encoding
        self.descriptor_extractor = descriptor_extractor
        self.iterative_fit = iterative_fit
        self.skip_clustering_fit = skip_clustering_fit

    @property
    def clustering(self):
        return self.encoding.clustering

    def fit(self, X, y=None):
        """
        Fit the clustering model.

        Parameters
        ----------
        X : array-like
            Either:
            - List of image paths (if descriptor_extractor is provided)
            - List of descriptor arrays
            - Single stacked array of all descriptors
        y : array-like, optional
            Target labels (unused, for sklearn compatibility)

        Returns
        -------
        self
        """
        if self.skip_clustering_fit:
            if not self.clustering._is_fitted:
                raise ValueError(
                    "skip_clustering_fit=True but clustering is not fitted! "
                    "Either fit the clustering first or set skip_clustering_fit=False"
                )
            print("Skipping clustering fit (using pre-fitted clusters)")
            return self

        if self.descriptor_extractor is not None:
            if self.iterative_fit:
                self.clustering.fit_iterative(X, load_func=self.descriptor_extractor)
            else:
                all_descriptors = []
                for path in X:
                    desc = self.descriptor_extractor(path)
                    if desc is not None and desc.size > 0:
                        all_descriptors.append(desc)
                if all_descriptors:
                    all_descriptors = np.vstack(all_descriptors)
                    self.clustering.fit(all_descriptors)
        else:
            if isinstance(X, list):
                X = np.vstack([d for d in X if d is not None and d.size > 0])
            self.clustering.fit(X)

        return self

    def transform(self, X):
        """
        Encode images/descriptors into fixed-length vectors.

        Parameters
        ----------
        X : array-like
            Either:
            - List of image paths (if descriptor_extractor is provided)
            - List of descriptor arrays (each image's descriptors)

        Returns
        -------
        np.ndarray
            Encoded vectors, shape (n_samples, encoding_dim)
        """
        if isinstance(X, str) or (hasattr(X, 'ndim') and X.ndim == 0):
            X = [X]
        elif hasattr(X, 'ndim') and X.ndim == 1 and X.dtype == object and len(X) == 0:
            pass
        elif not hasattr(X, '__iter__') or isinstance(X, (str, bytes)):
            X = [X]

        encoded = []

        for sample in X:
            descriptors = (self.descriptor_extractor(sample)
                        if self.descriptor_extractor is not None
                        else sample)
            encoded.append(self.encoding.encode(descriptors))

        return np.array(encoded)

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        params = {
            'clustering': self.clustering,
            'encoding': self.encoding,
            'descriptor_extractor': self.descriptor_extractor,
            'iterative_fit': self.iterative_fit
        }

        if deep and hasattr(self.clustering, 'get_params'):
            cluster_params = self.clustering.__dict__.copy()
            params.update({f'clustering__{k}': v for k, v in cluster_params.items()})

        return params

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            if '__' in key:
                obj_name, param_name = key.split('__', 1)
                obj = getattr(self, obj_name)
                setattr(obj, param_name, value)
            else:
                setattr(self, key, value)
        return self

    def to_dict(self):
        """Return a dictionary representation of the entire pipeline."""
        return {
            'type': self.__class__.__name__,
            'encoding': self.encoding.to_dict(),
            'iterative_fit': self.iterative_fit,
            'skip_clustering_fit': self.skip_clustering_fit,
            'has_descriptor_extractor': self.descriptor_extractor is not None
        }

    def __repr__(self):
        extractor_info = (
            f"{self.descriptor_extractor.__class__.__name__}"
            if hasattr(self.descriptor_extractor, '__class__')
            else type(self.descriptor_extractor).__name__
        ) if self.descriptor_extractor else None

        return (
            f"{self.__class__.__name__}(\n"
            f"  encoding={self.encoding!r},\n"
            f"  descriptor_extractor={extractor_info},\n"
            f"  iterative_fit={self.iterative_fit}\n"
            f")"
        )


# ============================================================================
# Convenience Factory Functions
# ============================================================================

def create_bovw_pipeline(
    n_clusters: int = 512,
    batch_size: int = 1024,
    normalize: bool = True,
    descriptor_extractor: Optional[DescriptorExtractor] = None,
    iterative_fit: bool = False,
    clustering: Optional[ClusteringAlgorithm] = None
) -> VisualEncodingTransformer:
    """
    Create a BoVW encoding pipeline.

    Example:
        >>> extractor = DescriptorExtractor(descriptor_type='SIFT', n_features=500)
        >>> bovw = create_bovw_pipeline(
        ...     n_clusters=1024,
        ...     descriptor_extractor=extractor.extract_from_file
        ... )
        >>> bovw.fit(train_image_paths)
    """
    if clustering is None:
        clustering = MiniBatchKMeansClustering(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=42
        )
        skip_fit = False
    else:
        skip_fit = clustering._is_fitted
        if skip_fit:
            print(f"Using pre-fitted clustering with {clustering.n_clusters} clusters")

    encoding = BagOfVisualWords(clustering, normalize_hist=normalize)

    extract_func = None
    if descriptor_extractor is not None:
        if isinstance(descriptor_extractor, DescriptorExtractor):
            extract_func = descriptor_extractor.extract_from_file
        else:
            extract_func = descriptor_extractor

    return VisualEncodingTransformer(
        encoding=encoding,
        descriptor_extractor=extract_func,
        iterative_fit=iterative_fit,
        skip_clustering_fit=skip_fit
    )


def create_vlad_pipeline(
    n_clusters: int = 256,
    batch_size: int = 1024,
    normalize: bool = True,
    descriptor_extractor: Optional[DescriptorExtractor] = None,
    iterative_fit: bool = False,
    clustering: Optional[ClusteringAlgorithm] = None
) -> VisualEncodingTransformer:
    """Create a VLAD encoding pipeline."""
    if clustering is None:
        clustering = MiniBatchKMeansClustering(
            n_clusters=n_clusters,
            batch_size=batch_size,
            random_state=42
        )
        skip_fit = False
    else:
        skip_fit = clustering._is_fitted
        if skip_fit:
            print(f"Using pre-fitted clustering with {clustering.n_clusters} clusters")

    encoding = VLAD(clustering, normalize_vlad=normalize)

    extract_func = None
    if descriptor_extractor is not None:
        if isinstance(descriptor_extractor, DescriptorExtractor):
            extract_func = descriptor_extractor.extract_from_file
        else:
            extract_func = descriptor_extractor

    return VisualEncodingTransformer(
        encoding=encoding,
        descriptor_extractor=extract_func,
        iterative_fit=iterative_fit,
        skip_clustering_fit=skip_fit
    )
