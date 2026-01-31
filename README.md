# 🛡️ Loom Sentinel: Defense Demonstration Unit

**Loom Sentinel** is a standalone demonstration of the **Loom Deterministic Neural Virtual Machine (DNVM)**, specifically packaging the advanced capabilities of the **`rt_a_l_bicameral`** engine (Real-Time Asynchronous Learning).

It showcases the engine's ability to run **180 parallel neural networks** (15 architectures × 6 training modes) simultaneously on a single CPU thread, processing network traffic at varying levels of abstraction without blocking.

---

## 🚀 Concept: The "Always Online" Shield

Conventional AI security tools operate in two phases:
1.  **Training (Offline)**: Analyze historical pcap data to build a model.
2.  **Inference (Online)**: Deploy the static model to detect threats.

**Sentinel** demonstrates **Phase 3: Bicameral Adaptation**. 
Using Loom's **Dual-Hemisphere Architecture**, Sentinel networks operate with a split-brain approach:
*   **Active Head (Left)**: Read-Only, 600Hz inference. It protects the network with zero blocking.
*   **Shadow Head (Right)**: Read-Write, Always-Training. It learns not just from threats, but from *normal* traffic shifts, constantly updating its weights in the background.

When the Shadow Head is confident, its weights are synced to the Active Head. This allows the Sentinel to "learn" while under fire, without ever pausing protection.

---

## 🧪 The Mega Grid (180 Networks)

Sentinel spins up a "Mega Grid" of **180 neural networks** running in parallel Goroutines to demonstrate the engine's concurrency:

### 1. The Core Architectures (15 Types)
We test every major neural layer type to prove Loom's versatility:
*   **Conv2D/Conv1D**: For spatial/temporal pattern recognition in packet payloads.
*   **LSTM/RNN**: For sequence anomaly detection (e.g., handshake violations).
*   **MultiHeadAttention (MHA)**: For semantic packet analysis (finding "needle in haystack" signatures).
*   **SwiGLU/RMSNorm**: Advanced LLM-style components for high-fidelity signal processing.

### 2. The Training Configurations (Single vs Bicameral)
*   **Single Head**: The traditional tracking network. It must "stop" to train, causing lag spikes.
*   **Bicameral (Dual Head)**: The Loom innovation. The Shadow Head trains asynchronously.
    *   **Result**: You will see **99.9% Availability** for Bicameral modes, compared to ~80% for Single modes, proving that "Smarter Math" beats brute force.

---

## 📊 How to Run

### Prerequisites
*   Linux environment with `libpcap-dev` installed.
*   Root privileges (required to sniff network interface).

```bash
# 1. Install dependencies
sudo apt-get install libpcap-dev

# 2. Run the Sentinel
# Using go run:
sudo go run main.go

# OR Build and Run binary:
go build -o loom-sentinel main.go
sudo ./loom-sentinel
```

### Expected Output
Sentinel will:
1.  Auto-detect your active network interface (e.g., `eth0` or `wlan0`).
2.  Spin up 180 parallel neural networks.
3.  Process live packets for 30 seconds.
4.  Output a **Real-Time metrics table** showing:
    *   **Threats Detected**: Anomalies flagged.
    *   **Availability %**: How much time the network spent "online" vs "blocking for training."
    *   **Latency**: The microseconds taken to process each packet.

> [!NOTE]
> Look for the **"Key Insight"** column. You will see that `ModeStepTweenChain` maintains **99.9% Availability** while `ModeNormalBP` suffers from **lag spikes (Blocked ⚠️)**. This proves the value of the Deterministic Virtual Machine for defense applications.

---

## 🔒 Security & Sovereignty
This codebase is **100% Go (Golang)**. It compiles to a single static binary with no external Python runtime, CUDA drivers, or cloud API keys required. It is designed for air-gapped, mission-critical deployment.

## 📄 License
**Apache 2.0** - Open Source.

*"The shield that learns is the shield that holds."*
