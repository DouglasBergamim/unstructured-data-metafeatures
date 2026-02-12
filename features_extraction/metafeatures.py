"""Meta-feature extraction using PyMFE."""

import numpy as np
import pandas as pd
from pymfe.mfe import MFE
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class MetaFeaturesExtractor:
    """Extract meta-features from feature matrices using PyMFE."""
    
    def __init__(self, random_state: int = 42):
        """Initialize meta-features extractor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        logger.debug(f"Initialized MetaFeaturesExtractor with random_state={random_state}")
    
    def extract(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Union[List[str], str] = "all",
        summaries: Optional[List[str]] = None,
        dataset_name: str = "unknown"
    ) -> pd.DataFrame:
        """Extract meta-features using PyMFE.
        
        Args:
            X: Feature matrix [N, D]
            y: Labels [N]
            groups: Meta-feature groups ("all" or list of group names)
            summaries: Summary functions (default: ["mean", "sd"])
            dataset_name: Dataset identifier
            
        Returns:
            DataFrame with columns: feature, value, group, dataset
            
        Example:
            >>> extractor = MetaFeaturesExtractor(random_state=42)
            >>> meta_df = extractor.extract(
            ...     X=features,
            ...     y=labels,
            ...     groups=["statistical", "model-based"],
            ...     dataset_name="my_dataset"
            ... )
        """
        logger.info(
            f"Extracting meta-features: X.shape={X.shape}, "
            f"y.shape={y.shape}, groups={groups}"
        )
        
        # Standardize features
        X_std = self.scaler.fit_transform(X)
        y = y.astype(int).ravel()
        
        # Compute optimal CV folds
        num_cv_folds = self._compute_cv_folds(y)
        logger.debug(f"Using {num_cv_folds} CV folds")
        
        # Validate and prepare summaries
        summaries = self._validate_summaries(summaries)
        logger.debug(f"Using summaries: {summaries}")
        
        # Determine group list
        group_list = self._prepare_groups(groups)
        logger.info(f"Extracting {len(group_list)} meta-feature groups")
        
        # Extract meta-features per group
        dfs: List[pd.DataFrame] = []
        
        if group_list:
            for group_name in group_list:
                df_group = self._extract_group(
                    X_std, y, group_name, summaries, num_cv_folds
                )
                if df_group is not None:
                    dfs.append(df_group)
        else:
            # Fallback: extract without explicit groups
            df_group = self._extract_no_groups(X_std, y, summaries, num_cv_folds)
            if df_group is not None:
                dfs.append(df_group)
        
        # Combine results
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            logger.warning("No meta-features extracted")
            df = pd.DataFrame(columns=["feature", "value", "group"])
        
        df["dataset"] = dataset_name
        logger.info(f"Extracted {len(df)} meta-features")
        
        return df
    
    def _compute_cv_folds(self, y: np.ndarray) -> int:
        """Compute optimal CV folds based on class distribution.
        
        Args:
            y: Label array
            
        Returns:
            Number of CV folds (2-3)
        """
        _, counts = np.unique(y, return_counts=True)
        
        if counts.size == 0:
            return 2
        
        min_class_count = counts.min()
        
        if counts.size >= 2:
            # At least 2 classes: use min class count to determine folds
            return max(2, min(3, int(min_class_count)))
        else:
            # Single class edge case
            return 2
    
    def _validate_summaries(self, summaries: Optional[List[str]]) -> Optional[List[str]]:
        """Validate and return valid summary functions.

        Args:
            summaries: Requested summary functions or None for no summarization

        Returns:
            List of valid summary function names, or None for raw values
        """
        # If None, return None to get raw metafeature values
        if summaries is None:
            return None

        try:
            valid_summaries = set(MFE.valid_summary())
        except Exception:
            valid_summaries = {"mean", "sd", "median", "min", "max"}

        if not summaries:
            return ["mean", "sd"]

        # Filter to valid summaries
        validated = [s for s in summaries if s in valid_summaries]

        if not validated:
            logger.warning(
                f"No valid summaries in {summaries}, "
                f"using default ['mean', 'sd']"
            )
            return ["mean", "sd"]

        return validated
    
    def _prepare_groups(self, groups: Union[List[str], str]) -> List[str]:
        """Prepare list of meta-feature groups.
        
        Args:
            groups: "all" or list of group names
            
        Returns:
            List of group names to extract
        """
        if groups is None or (isinstance(groups, str) and groups.lower() == "all"):
            try:
                return list(MFE.valid_groups())
            except Exception as e:
                logger.warning(f"Could not get valid groups: {e}")
                return []
        
        if isinstance(groups, (list, tuple, set)):
            return list(groups)
        
        return [groups]
    
    def _extract_group(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_name: str,
        summaries: List[str],
        num_cv_folds: int
    ) -> Optional[pd.DataFrame]:
        """Extract meta-features for a specific group.
        
        Args:
            X: Standardized features
            y: Labels
            group_name: Name of the meta-feature group
            summaries: Summary functions
            num_cv_folds: Number of CV folds
            
        Returns:
            DataFrame with meta-features or None if extraction failed
        """
        try:
            # Try different MFE constructor signatures for compatibility
            mfe = self._create_mfe(group_name, summaries, num_cv_folds)
            mfe.fit(X, y)
            names, values = mfe.extract()

            df = self._normalize_mfe_output(names, values)
            df["group"] = group_name

            logger.info(f"Extracted {len(names)} features for group '{group_name}'")
            return df
            
        except Exception as e:
            logger.warning(f"Skipping group '{group_name}' due to error: {e}")
            return None
    
    def _extract_no_groups(
        self,
        X: np.ndarray,
        y: np.ndarray,
        summaries: List[str],
        num_cv_folds: int
    ) -> Optional[pd.DataFrame]:
        """Extract meta-features without explicit groups.
        
        Args:
            X: Standardized features
            y: Labels
            summaries: Summary functions
            num_cv_folds: Number of CV folds
            
        Returns:
            DataFrame with meta-features or None if extraction failed
        """
        try:
            mfe = self._create_mfe(None, summaries, num_cv_folds)
            mfe.fit(X, y)
            names, values = mfe.extract()

            df = self._normalize_mfe_output(names, values)
            df["group"] = "unknown"

            logger.info(f"Extracted {len(names)} features (no explicit groups)")
            return df
            
        except Exception as e:
            logger.warning(f"Meta-feature extraction failed: {e}")
            return None
    
    @staticmethod
    def _normalize_mfe_output(names: list, values: list) -> pd.DataFrame:
        """Normalize MFE output into a flat DataFrame with scalar values.

        When ``summaries=None`` PyMFE returns raw per-sample arrays instead of
        scalars.  This method explodes those arrays so that every row in the
        resulting DataFrame has a single float in the ``value`` column, making
        it compatible with Parquet serialisation.

        Args:
            names: Feature names returned by ``mfe.extract()``.
            values: Corresponding values (scalars or arrays).

        Returns:
            DataFrame with columns ``feature``, ``value``, and ``index``.
            ``index`` is 0 for scalar features; 0…N-1 for exploded arrays.
        """
        rows: list = []
        for name, val in zip(names, values):
            if isinstance(val, np.ndarray):
                for i, v in enumerate(val.ravel()):
                    rows.append({"feature": name, "value": float(v), "index": i})
            else:
                rows.append({"feature": name, "value": float(val), "index": 0})
        return pd.DataFrame(rows)

    def _create_mfe(
        self,
        group_name: Optional[str],
        summaries: List[str],
        num_cv_folds: int
    ) -> MFE:
        """Create MFE instance with version compatibility handling.
        
        Args:
            group_name: Group name (or None for all groups)
            summaries: Summary functions
            num_cv_folds: Number of CV folds
            
        Returns:
            Configured MFE instance
        """
        base_kwargs = {
            "summary": summaries,
            "random_state": self.random_state,
            "score": "accuracy",
            "num_cv_folds": num_cv_folds,
            "lm_sample_frac": 0.5,
        }
        
        if group_name is not None:
            # Try list format first
            try:
                return MFE(groups=[group_name], shuffle_cv_folds=True, **base_kwargs)
            except TypeError:
                # Try string format
                try:
                    return MFE(groups=group_name, shuffle_cv_folds=True, **base_kwargs)
                except TypeError:
                    # Older version without shuffle_cv_folds
                    return MFE(groups=group_name, **base_kwargs)
        else:
            # No groups specified
            try:
                return MFE(shuffle_cv_folds=True, **base_kwargs)
            except TypeError:
                return MFE(**base_kwargs)
