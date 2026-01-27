"""
Dataset preprocessing utilities for PCL network input.
"""

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from src.events.Events import Events


def dvsGesture128_toPCL(events_path, width, height):
    """
    Load and preprocess DVS-Gesture128 events for PCL network input.
    
    Scales down the events from the original 128x128 dimensions to the target
    dimensions and centers them in the output.
    
    Args:
        events_path (str): Path to the .npz file containing the events
        width (int): Target width for the PCL network (e.g., 66)
        height (int): Target height for the PCL network (e.g., 66)
    
    Returns:
        Events: Events object with scaled and centered event_array
    """
    # Load events
    events = Events(events_path)
    
    # Calculate bounds
    max_x_orig = max(events.event_array["x"])
    max_y_orig = max(events.event_array["y"])
    min_x_orig = min(events.event_array["x"])
    min_y_orig = min(events.event_array["y"])
    
    # Original DVS-Gesture128 dimensions (use actual range)
    original_width = max_x_orig - min_x_orig + 1
    original_height = max_y_orig - min_y_orig + 1
    
    # Calculate scaling factors
    scale_x = original_width / width
    scale_y = original_height / height
    
    # First, translate events to start at (0, 0)
    events.event_array["x"] = (events.event_array["x"] - min_x_orig).astype("<u2")
    events.event_array["y"] = (events.event_array["y"] - min_y_orig).astype("<u2")
    
    # Scale down to target dimensions
    events.event_array["x"] = (events.event_array["x"] / scale_x).astype("<u2")
    events.event_array["y"] = (events.event_array["y"] / scale_y).astype("<u2")
    
    # Recalculate after scaling
    max_x = max(events.event_array["x"])
    max_y = max(events.event_array["y"])
    min_x = min(events.event_array["x"])
    min_y = min(events.event_array["y"])
    
    # Calculate actual center of scaled events
    actual_center_x = (max_x + min_x) / 2
    actual_center_y = (max_y + min_y) / 2
    
    # Desired center in output dimensions
    desired_center_x = width / 2
    desired_center_y = height / 2
    
    # Calculate shift to center the scaled events
    shift_x = actual_center_x - desired_center_x
    shift_y = actual_center_y - desired_center_y
    
    # Apply centering shift to event coordinates
    events.event_array["x"] = (events.event_array["x"] - shift_x).astype("<u2")
    events.event_array["y"] = (events.event_array["y"] - shift_y).astype("<u2")
    
    return events


def natural_images_toPCL(events_path, width, height, original_width=None, original_height=None):
    """
    Load and preprocess Natural Images events for PCL network input.
    
    Scales down the events from the original dimensions to the target
    dimensions and centers them in the output. If original dimensions are not
    provided, they are automatically detected from the event data range.
    
    Args:
        events_path (str): Path to the .npz file containing the events
        width (int): Target width for the PCL network (e.g., 66)
        height (int): Target height for the PCL network (e.g., 66)
        original_width (int, optional): Original width of the natural images dataset.
                                        If None, auto-detected from data range (max-min+1).
        original_height (int, optional): Original height of the natural images dataset.
                                         If None, auto-detected from data range (max-min+1).
    
    Returns:
        Events: Events object with scaled and centered event_array
    """
    # Load events
    events = Events(events_path)
    
    # Calculate bounds
    max_x_orig = max(events.event_array["x"])
    max_y_orig = max(events.event_array["y"])
    min_x_orig = min(events.event_array["x"])
    min_y_orig = min(events.event_array["y"])
    
    # Auto-detect original dimensions if not provided (use range, not max)
    if original_width is None:
        original_width = max_x_orig - min_x_orig + 1
    if original_height is None:
        original_height = max_y_orig - min_y_orig + 1
    
    # Calculate scaling factors
    scale_x = original_width / width
    scale_y = original_height / height
    
    # First, translate events to start at (0, 0)
    events.event_array["x"] = (events.event_array["x"] - min_x_orig).astype("<u2")
    events.event_array["y"] = (events.event_array["y"] - min_y_orig).astype("<u2")
    
    # Calculate bounds after translation (for centering later)
    max_x = max(events.event_array["x"])
    max_y = max(events.event_array["y"])
    min_x = min(events.event_array["x"])
    min_y = min(events.event_array["y"])
    
    # Scale down to target dimensions
    events.event_array["x"] = (events.event_array["x"] / scale_x).astype("<u2")
    events.event_array["y"] = (events.event_array["y"] / scale_y).astype("<u2")
    
    # Recalculate after scaling
    max_x = max(events.event_array["x"])
    max_y = max(events.event_array["y"])
    min_x = min(events.event_array["x"])
    min_y = min(events.event_array["y"])
    
    # Calculate actual center of scaled events
    actual_center_x = (max_x + min_x) / 2
    actual_center_y = (max_y + min_y) / 2
    
    # Desired center in output dimensions
    desired_center_x = width / 2
    desired_center_y = height / 2
    
    # Calculate shift to center the scaled events
    shift_x = actual_center_x - desired_center_x
    shift_y = actual_center_y - desired_center_y
    
    # Apply centering shift to event coordinates
    events.event_array["x"] = (events.event_array["x"] - shift_x).astype("<u2")
    events.event_array["y"] = (events.event_array["y"] - shift_y).astype("<u2")
    
    return events


def concatenate_npz_to_single_npz(input_dir, output_file, time_spacing=1000):
    """
    Concatenate multiple .npz event files into a single .npz file.
    
    Events from different files are concatenated chronologically, with
    configurable time spacing between files to maintain temporal separation.
    
    Args:
        input_dir (str or Path): Directory containing .npz files to concatenate
        output_file (str or Path): Output path for concatenated .npz file
        time_spacing (int): Microseconds to add between consecutive files (default: 1000 μs = 1 ms)
    
    Returns:
        dict: Statistics about the concatenation (num_files, total_events, time_range)
    """
    input_dir = Path(input_dir)
    npz_files = sorted(input_dir.glob("*.npz"))
    
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {input_dir}")
    
    print(f"Concatenating {len(npz_files)} files...")
    
    all_events = []
    current_time_offset = 0
    
    for npz_file in tqdm(npz_files):
        # Load events
        events = Events(str(npz_file))
        event_array = events.event_array.copy()
        
        # Add time offset to maintain chronological order
        if len(event_array) > 0:
            event_array["t"] = event_array["t"] + current_time_offset
            all_events.append(event_array)
            
            # Update offset for next file (max time + spacing)
            current_time_offset = event_array["t"].max() + time_spacing
    
    # Concatenate all events
    concatenated_events = np.concatenate(all_events)
    
    # Sort by timestamp to ensure chronological order
    concatenated_events = np.sort(concatenated_events, order='t')
    
    # Save as .npz
    np.savez(output_file, events=concatenated_events)
    
    stats = {
        'num_files': len(npz_files),
        'total_events': len(concatenated_events),
        'time_range': (concatenated_events['t'].min(), concatenated_events['t'].max())
    }
    
    print(f"\n✓ Concatenated {stats['num_files']} files into {output_file}")
    print(f"  Total events: {stats['total_events']:,}")
    print(f"  Time range: {stats['time_range'][0]} - {stats['time_range'][1]} μs")
    
    return stats


def concatenate_npz_to_h5(input_dir, output_file, dataset_name='events', normalize_time=True):
    """
    Concatenate multiple .npz event files into a single HDF5 (.h5) file.
    
    HDF5 format is more efficient for large datasets and is the standard
    format used in neuromorphic computing. Events are stored as a structured
    array with efficient compression.
    
    Args:
        input_dir (str or Path): Directory containing .npz files to concatenate
        output_file (str or Path): Output path for .h5 file
        dataset_name (str): Name of the dataset in the HDF5 file (default: 'events')
        normalize_time (bool): If True, reset timestamps to start at 0 (default: True)
    
    Returns:
        dict: Statistics about the concatenation (num_files, total_events, time_range)
    """
    input_dir = Path(input_dir)
    npz_files = sorted(input_dir.glob("*.npz"))
    
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {input_dir}")
    
    print(f"Concatenating {len(npz_files)} files to HDF5...")
    
    all_events = []
    current_time_offset = 0
    
    for npz_file in tqdm(npz_files):
        # Load events
        events = Events(str(npz_file))
        event_array = events.event_array.copy()
        
        # Add time offset to maintain chronological order
        if len(event_array) > 0:
            # Normalize each file's timestamps to start at 0 if normalize_time is True
            if normalize_time:
                min_t = event_array["t"].min()
                event_array["t"] = event_array["t"] - min_t
            
            # Add offset to continue from previous file
            event_array["t"] = event_array["t"] + current_time_offset
            all_events.append(event_array)
            
            # Update offset for next file (continue from max time)
            current_time_offset = event_array["t"].max()
    
    # Concatenate all events
    concatenated_events = np.concatenate(all_events)
    
    # Sort by timestamp to ensure chronological order
    concatenated_events = np.sort(concatenated_events, order='t')
    
    # Delete existing file if it exists to avoid append mode issues
    output_file = Path(output_file)
    if output_file.exists():
        output_file.unlink()
    
    # Create Events object with concatenated data and use built-in save method
    final_events = Events(concatenated_events)
    final_events.save_as_file(str(output_file))
    
    stats = {
        'num_files': len(npz_files),
        'total_events': len(concatenated_events),
        'time_range': (int(concatenated_events['t'].min()), int(concatenated_events['t'].max())),
        'file_size_mb': Path(output_file).stat().st_size / (1024 * 1024)
    }
    
    print(f"\n✓ Concatenated {stats['num_files']} files into {output_file}")
    print(f"  Total events: {stats['total_events']:,}")
    print(f"  Time range: {stats['time_range'][0]} - {stats['time_range'][1]} μs")
    print(f"  File size: {stats['file_size_mb']:.2f} MB")
    
    return stats
