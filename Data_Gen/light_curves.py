from astroquery.mast import Observations
import lightkurve as lk
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np 

VIEW = True
GEN = False


def compute_quiet_curve(light_curve_path):
    """
    Check if a light curve is quiet (low variability).
    
    Args:
        light_curve_path: Path to FITS file
        
    Returns:
        (lc_object, variability) if quiet, else None
    """
    try:
        lc = lk.read(light_curve_path)
        
        # Normalize and filter (same as your preprocessing)
        lc = lc.remove_nans().normalize()
        mask = (lc.quality <= 2)
        lc_filtered = lc[mask]
        
        if len(lc_filtered.flux) < 100:
            return None
        
        # Get flux values (this is what you were missing!)
        flux = lc_filtered.flux.value
        
        # Handle MaskedArrays
        if hasattr(flux, 'filled'):
            flux = flux.filled(np.nan)
        flux = flux[~np.isnan(flux)]
        
        if len(flux) < 100:
            return None
        
        # Compute variability
        variability = np.std(flux) / np.mean(flux)
        
        # Check if quiet (< 1% variability)
        if variability < 0.01:
            return lc_filtered, variability
        else:
            return None
            
    except Exception as e:
        return None


if GEN:
    # Getting Injection stars from TESS

    # --- PARAMETERS ---
    N_LIGHTCURVES = 50   # how many quiet light curves you want

    quiet_baselines = []

    print("Querying TESS Sector 18...")
    
    # --- 1. Query TESS Sector 18 light curves ---
    obs_table = Observations.query_criteria(
        obs_collection="TESS",
        dataproduct_type="timeseries",
        sequence_number=18,
        t_exptime=120  # 2-minute cadence only
    )
    
    print(f"Found {len(obs_table)} observations in Sector 18")

    # --- 2. Get product list and filter for light curves ---
    print("Getting product list...")
    products = Observations.get_product_list(obs_table)
    lc_products = Observations.filter_products(
        products, 
        productSubGroupDescription="LC"
    )
    
    print(f"Found {len(lc_products)} light curve products")

    # --- 3. Download and check for quiet ones ---
    print(f"\nSearching for {N_LIGHTCURVES} quiet light curves...")
    print("Downloading in batches and checking variability...\n")
    
    batch_size = 50  # Download 50 at a time
    checked = 0
    
    for start_idx in range(0, len(lc_products), batch_size):
        if len(quiet_baselines) >= N_LIGHTCURVES:
            break
        
        # Download batch
        end_idx = min(start_idx + batch_size, len(lc_products))
        print(f"Downloading batch {start_idx//batch_size + 1} ({start_idx+1}-{end_idx})...")
        
        manifest = Observations.download_products(
            lc_products[start_idx:end_idx],
            download_dir='temp_injection_lcs'
        )
        
        # Check each downloaded file
        for row in manifest:
            if len(quiet_baselines) >= N_LIGHTCURVES:
                break
            
            if row['Status'] != 'COMPLETE':
                continue
            
            checked += 1
            
            # Check if quiet
            result = compute_quiet_curve(row['Local Path'])
            
            if result is not None:
                lc_filtered, variability = result
                
                # Extract normalized flux array
                flux = lc_filtered.flux.value
                if hasattr(flux, 'filled'):
                    flux = flux.filled(np.nan)
                flux = flux[~np.isnan(flux)]
                flux_normalized = flux.astype(np.float32)
                
                quiet_baselines.append({
                    'flux': flux_normalized,
                    'variability': variability,
                    'lc_object': lc_filtered,
                    'path': row['Local Path']
                })
                
                print(f"  ✓ Found quiet LC {len(quiet_baselines)}/{N_LIGHTCURVES}: "
                      f"variability = {variability:.6f}")
        
        print(f"  Checked {checked} light curves so far, "
              f"found {len(quiet_baselines)} quiet ones\n")
    
    print(f"\n{'='*60}")
    print(f"FOUND {len(quiet_baselines)} QUIET LIGHT CURVES")
    print(f"{'='*60}")
    
    for i, baseline in enumerate(quiet_baselines):
        print(f"Baseline {i+1}: variability = {baseline['variability']:.6f}, "
              f"length = {len(baseline['flux'])}")
    
    # --- 4. Save quiet baselines ---
    save_path = Path('Data_Gen/injection_sets/quiet_baselines_sector18.npz')
    
    save_dict = {
        f'baseline_{i}': baseline['flux'] 
        for i, baseline in enumerate(quiet_baselines)
    }
    
    # Also save metadata
    metadata = {
        'variabilities': [b['variability'] for b in quiet_baselines],
        'paths': [str(b['path']) for b in quiet_baselines],
    }
    
    np.savez_compressed(save_path, **save_dict)
    np.save('Data_Gen/injection_sets/quiet_baselines_metadata.npy', metadata)
    
    print(f"\n✓ Saved to {save_path}")
    print(f"✓ Metadata saved to Data_Gen/injection_sets/quiet_baselines_metadata.npy")


if VIEW:
    # --- 5. Optional: Visualize all dyson positives ---
    print("\nVisualizing dyson positive light curves...")
    
    dyson_dir = 'results/thin'
    import os
    
    # Get all .npz files in the directory
    npz_files = [f for f in os.listdir(dyson_dir) if f.endswith('.npz')]
    npz_files.sort()  # Sort files for consistent ordering
    
    for file_name in npz_files:
        file_path = os.path.join(dyson_dir, file_name)
        data = np.load(file_path)
        
        # Get curve number from filename
        curve_num = file_name.split('.')[0]
        
        # Create a new figure for each light curve
        plt.figure(figsize=(12, 4))
        
        times = data['times'] if 'times' in data else np.arange(len(data['flux']))
        flux = data['flux']
        
        plt.plot(times, flux, linewidth=0.5, alpha=0.7)
        plt.title(f"Dyson Positive Light Curve - {curve_num}")
        plt.ylabel("Normalized Flux")
        plt.xlabel("Time (days)")
        plt.tight_layout()
        plt.show()
        print(f"times {times} \n fluxes:{flux}")
        
    print(f"✓ Displayed {len(npz_files)} dyson positive light curves")