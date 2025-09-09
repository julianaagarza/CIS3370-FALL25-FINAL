# Final Project for intrusion Detection Systems Fall 2025

## Team Members:
- Sarah Steadham
- Juliana Garza

## Overview
This project focuses on detecting malware hidden in encrypted network traffic, with a particular focus on comparing TLS and QUIC protocols. The goal is to integrate machine learning with an IDS workflow (Suricata + ELK stack) to detect malicious flows based on metadata and flow-level features rather than decrypted payloads. Additionally, the project will evaluate whether malware behaves differently across TLS and QUIC, highlighting areas where traditional IDS may struggle.

## Key Features/Unique Spin
- **Protocol Comparison (QUIC vs TLS)** – Investigate whether malware is easier or harder to detect depending on the encrypted protocol.
- **Explainable ML** – Use SHAP or LIME to show which flow features contribute most to detecting malicious activity.
- **IDS Integration** – Incorporate Suricata for real-time log generation and ELK for visualization of alerts, bridging theory with practical security operations.
- **Flow-Based Analysis** – Focus on packet timing, flow duration, handshake metadata, and packet size distributions, not payload content.
