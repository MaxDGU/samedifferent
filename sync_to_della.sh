        #!/bin/bash
        
        # --- Configuration ---
        DELLA_USER="mg7411" # Replace with your Della NetID
        DELLA_HOST="della.princeton.edu"
        # Base directory on Della where your flat structure lives
        DELLA_PROJECT_BASE="/scratch/gpfs/${DELLA_USER}"
        
        # --- Files/Dirs to Sync (Local Path -> Della Path segment if different) ---
        # Add more entries as needed.
        # This example assumes your local structure is like:
        # ./remote/naturalistic/meta_train/meta_naturalistic_train.py -> $DELLA_PROJECT_BASE/meta_naturalistic_train.py
        # ./conv2.py -> $DELLA_PROJECT_BASE/conv2.py
        # ./remote/naturalistic/run_meta_naturalistic_della.sh -> $DELLA_PROJECT_BASE/run_meta_naturalistic_della.sh
        
        echo "Syncing files to Della (${DELLA_USER}@${DELLA_HOST}:${DELLA_PROJECT_BASE})..."
        
        # Training script
        rsync -avz --progress ./remote/naturalistic/meta_train/meta_naturalistic_train.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_PROJECT_BASE}/meta_naturalistic_train.py
        
        # Model files (assuming they are at the root of your local project or a known relative path)
        # If models are in, e.g., ./models/ then use ./models/conv2.py
        rsync -avz --progress ./baselines/models/conv2.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_PROJECT_BASE}/conv2.py
        rsync -avz --progress ./baselines/models/conv4.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_PROJECT_BASE}/conv4.py
        rsync -avz --progress ./baselines/models/conv6.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_PROJECT_BASE}/conv6.py
        
        # Slurm script
        rsync -avz --progress ./remote/naturalistic/run_meta_naturalistic_della.sh ${DELLA_USER}@${DELLA_HOST}:${DELLA_PROJECT_BASE}/run_meta_naturalistic_della.sh

        # Example for analysis scripts if they also run on Della with a flat structure
        # rsync -avz --progress ./remote/baselines/analyze_pb_weights.py ${DELLA_USER}@${DELLA_HOST}:${DELLA_PROJECT_BASE}/analyze_pb_weights.py
        
        echo "Sync complete."