import numpy as np
import random

class ReplayBuffer(object):
    """Standard replay buffer for reinforcement learning.
    
    Stores transitions and provides uniform sampling for training.
    """
    
    def __init__(self, size, env_dict):
        """Initialize replay buffer.

        Parameters
        ----------
        size: int
            Maximum number of transitions to store. When full, old memories are dropped.
        env_dict: dict
            Dictionary specifying shape and type of all buffer elements.
        """
        # Core buffer storage
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._env_dict = env_dict

    def __len__(self):
        return len(self._storage)

    def add(self, data):
        """Add a new entry to the buffer. Ensure data matches the expected types and shapes as per env_dict."""
        processed_data = self._process_data(data)
        self._store_data(processed_data)

    def _process_data(self, data):
        """Process and validate data according to environment dictionary."""
        processed_data = {}
        
        for key, value in data.items():
            expected_dtype = self._env_dict[key]['dtype']
            expected_shape = self._env_dict[key]['shape']

            # Pre-validation to catch NaNs or Infs
            if np.isnan(value).any() or np.isinf(value).any():
                print(f"Pre-add check: NaN or Inf detected in {key} with value {value}")

            # Convert and reshape data
            if isinstance(value, np.ndarray):
                value = value.astype(expected_dtype)
            else:
                value = np.array([value], dtype=expected_dtype)

            # Ensure correct shape
            if value.shape != expected_shape:
                value = np.reshape(value, expected_shape)

            processed_data[key] = value

            # Post-validation
            if np.isnan(value).any() or np.isinf(value).any():
                print(f"Post-add check: NaN or Inf detected in buffer for key '{key}': {value}")

        return processed_data

    def _store_data(self, processed_data):
        """Store processed data in the buffer."""
        if self._next_idx >= len(self._storage):
            self._storage.append(processed_data)
        else:
            self._storage[self._next_idx] = processed_data

        self._next_idx = int((self._next_idx + 1) % self._maxsize)

    def _encode_sample(self, idxes):
        batch_data = {key: [] for key in self._env_dict}
        for i in idxes:
            data = self._storage[i]
            for key in self._env_dict:
                batch_data[key].append(np.array(data[key], copy=False))
        return {key: np.array(batch_data[key]) for key in self._env_dict}

    def sample(self, batch_size):
        """Sample a batch of experiences uniformly.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        A dictionary of numpy arrays for each part of the experience batch.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

    def clear(self):
        """Clears the replay buffer of all stored data."""
        self._storage.clear()
        self._next_idx = 0