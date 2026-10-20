# Copy to scripts/local.sh (not in git) and fill in this setup's values; scripts/config.sh reads it first.
WORKER=chlee@192.168.50.155          # worker SSH target on the LAN (key-based authentication)
FABRIC_PEER=192.168.100.11           # worker CX-7 address used for rank traffic
# A third Spark (./start-tp3.sh, experimental; README "3 Sparks"):
# WORKER2=user@192.0.2.3       # rank 2
# WORKER_WEIGHTS2=nfs          # per worker: copy or nfs (default: WORKER_WEIGHTS)
# WORKER_HF_CACHE2=/models/hf   # per worker: its Hugging Face cache when it is not its HF_HOME
# ABLIT=1                      # the gated Ablit weights instead (README "Ablit weights"); needs HF_TOKEN below
# HF_TOKEN=hf_...              # a Hugging Face read token whose account agreed to the Ablit model's terms
