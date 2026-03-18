import io
import numpy as np
from cryptography.fernet import Fernet


class AdapterEncryption:
    """
    Symmetric encryption for LoRA adapter delta transmission.

    Each federated client owns one instance with its own randomly generated key.
    Parameters are serialized to bytes, encrypted with AES-128-CBC + HMAC-SHA256
    (via Fernet), and decrypted by the server before aggregation.

    This ensures that adapter deltas on the wire are always ciphertext - raw
    weight values are never transmitted in plaintext, even inside the simulation.

    In a real distributed deployment the client key would be shared with the
    aggregation server via a secure key-exchange protocol (e.g. ECDH). Here we
    simulate that by having each client expose its key so the server can decrypt.
    """

    def __init__(self) -> None:
        self._key = Fernet.generate_key()
        self._fernet = Fernet(self._key)

    # ------------------------------------------------------------------
    # Server-side helpers (server calls these with the client's key)
    # ------------------------------------------------------------------

    @classmethod
    def from_key(cls, key: bytes) -> "AdapterEncryption":
        """Reconstruct an instance from an existing key (used by the server)."""
        instance = object.__new__(cls)
        instance._key = key
        instance._fernet = Fernet(key)
        return instance

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize(params: list[np.ndarray]) -> bytes:
        buf = io.BytesIO()
        np.save(buf, len(params))
        for arr in params:
            np.save(buf, arr)
        return buf.getvalue()

    @staticmethod
    def _deserialize(data: bytes) -> list[np.ndarray]:
        buf = io.BytesIO(data)
        n = int(np.load(buf, allow_pickle=False))
        return [np.load(buf, allow_pickle=False) for _ in range(n)]

    # ------------------------------------------------------------------
    # Encrypt / decrypt
    # ------------------------------------------------------------------

    def encrypt(self, params: list[np.ndarray]) -> bytes:
        """Serialize and encrypt a list of numpy arrays."""
        return self._fernet.encrypt(self._serialize(params))

    def decrypt(self, payload: bytes) -> list[np.ndarray]:
        """Decrypt and deserialize a list of numpy arrays."""
        return self._deserialize(self._fernet.decrypt(payload))

    @property
    def key(self) -> bytes:
        return self._key
