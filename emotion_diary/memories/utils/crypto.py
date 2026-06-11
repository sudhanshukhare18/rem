import base64
import logging
from typing import Optional

from memories.encryption import decrypt_aes


logger = logging.getLogger(__name__)


MIN_AES_BYTES = 16


def is_non_empty_string(
    value
) -> bool:
    """
    Validate non-empty string input.
    """

    return (
        isinstance(value, str)
        and bool(value.strip())
    )


def looks_like_aes_ciphertext(
    value: str
) -> bool:
    """
    Check whether a value appears to be
    AES-encrypted Base64 ciphertext.
    """

    if not is_non_empty_string(value):
        return False

    # Base64 AES ciphertexts are usually longer
    if len(value.strip()) < 24:
        return False

    try:

        decoded = base64.b64decode(
            value,
            validate=True
        )

        # Very small decoded payloads
        # are unlikely to be AES ciphertext
        return len(decoded) >= MIN_AES_BYTES

    except Exception:
        return False


def safe_decrypt(
    value: Optional[str],
    fallback: str = ""
) -> str:
    """
    Safely decrypt AES-encrypted values.

    Returns:
        - decrypted text if valid AES
        - original value if plaintext
        - fallback if decryption fails
    """

    # Handle invalid input
    if value is None:
        return fallback

    # Non-string values
    if not isinstance(value, str):
        return fallback

    cleaned_value = value.strip()

    if not cleaned_value:
        return fallback

    try:

        # Decrypt only if ciphertext
        if looks_like_aes_ciphertext(
            cleaned_value
        ):

            decrypted = decrypt_aes(
                cleaned_value
            )

            # Protect against empty decrypts
            if (
                decrypted
                and isinstance(
                    decrypted,
                    str
                )
            ):
                return decrypted.strip()

            return fallback

        # Already plaintext
        return cleaned_value

    except Exception as error:

        logger.exception(
            "AES decryption failed: %s",
            error
        )

        return fallback