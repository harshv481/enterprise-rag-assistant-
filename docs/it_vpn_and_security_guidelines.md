# SEC-GUIDE-201: Enterprise IT Security, Endpoint Protection & VPN Access

## 1. Zero Trust Network Architecture & GlobalProtect VPN
Remote team members accessing internal production databases, staging environments, or internal intranets must authenticate through GlobalProtect VPN with YubiKey hardware token or Okta Verify Push with biometric number matching. Split tunneling is strictly disabled for all sensitive domain ranges (*.corp.internal).

## 2. Password and Credential Policy
All enterprise accounts require minimum 16-character passphrases with mandatory multi-factor authentication (MFA). 1Password Enterprise is provided to all employees for team vault credential sharing; master passwords must never be stored on physical media or unencrypted notes.

## 3. Laptop Encryption and Device Hygiene
Company-managed laptops (macOS FileVault 2 / Windows BitLocker) run continuous CrowdStrike Falcon EDR telemetry. Operating system updates must be installed within 7 calendar days of release. Unattended laptops in public locations must be locked using Windows Key + L or Command + Control + Q.
