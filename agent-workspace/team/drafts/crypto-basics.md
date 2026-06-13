TL;DR: Symmetric encryption uses one shared key for both encryption and decryption; asymmetric encryption uses a mathematically linked public/private key pair.

## Symmetric vs. Asymmetric Encryption

Symmetric encryption uses a single secret key to both encrypt and decrypt data. Both parties must possess the same key, making it fast and efficient for bulk data — AES is the dominant modern standard. The core challenge is key distribution: securely sharing that secret key over an untrusted network is non-trivial.

Asymmetric encryption solves the distribution problem by using two linked keys: a public key (freely shared) to encrypt, and a private key (kept secret) to decrypt. RSA and elliptic-curve cryptography (ECC) are common examples. The trade-off is computational cost — asymmetric operations are significantly slower than symmetric ones.

In practice, most secure protocols (TLS, SSH) combine both: asymmetric encryption establishes a shared session key, then symmetric encryption handles the bulk data transfer.

## Sources

- https://csrc.nist.gov/glossary/term/symmetric_key_cryptography
- https://csrc.nist.gov/glossary/term/asymmetric_cryptography
- https://www.cloudflare.com/learning/ssl/what-is-asymmetric-encryption/
