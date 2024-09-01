import marimo

__generated_with = "0.8.7"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import kaggle
    import zipfile
    import os
    import pandas as pd
    import shutil
    import pydicom
    import numpy as np
    import zarr
    import requests
    return kaggle, mo, np, os, pd, pydicom, requests, shutil, zarr, zipfile


@app.cell
def __():
    raw_data_path = "/cbica/home/gangarav/raw_data"
    # titles of the datasets when downloaded
    RSNA_2024_competition = 'rsna-2024-lumbar-spine-degenerative-classification'
    LSD_dataset = 'LSD'
    # places to put the processed data
    RSNA_processed = '/cbica/home/gangarav/RSNA_PROCESSED'
    LSD_processed = '/cbica/home/gangarav/LSD_PROCESSED'
    return (
        LSD_dataset,
        LSD_processed,
        RSNA_2024_competition,
        RSNA_processed,
        raw_data_path,
    )


@app.cell
def __(mo):
    mo.md(r"""# Utils""")
    return


@app.cell
def __(os, requests, shutil, zipfile):
    def delete_folder(path):
        # Check if the folder exists
        if os.path.exists(path):
            # Remove the folder and all its content
            shutil.rmtree(path)
            print("Folder deleted successfully.")
        else:
            print("Folder does not exist.")

    def delete_file(file_path):
        # Check if the file exists
        if os.path.exists(file_path):
            # Delete the file
            os.unlink(file_path)
            print(f"File {file_path} has been deleted.")
        else:
            print("The file does not exist.")

    def rename_folder(old_path, new_path):
        # Check if the current folder exists
        if os.path.exists(old_path):
            # Rename the folder
            os.rename(old_path, new_path)
            print("Folder renamed successfully.")
        else:
            print("The original folder does not exist.")

    def download_file(url, save_path):
        # Send a GET request to the URL
        response = requests.get(url, stream=True)

        # Open the file at the path you want to save it
        with open(save_path, 'wb') as f:
            # Write the content to the file in chunks
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File has been downloaded and saved to {save_path}")

    def unzip_file(zip_path, extract_to):
        # Open the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract all the contents into the directory
            zip_ref.extractall(extract_to)
            print(f"All files have been extracted to {extract_to}")
    return (
        delete_file,
        delete_folder,
        download_file,
        rename_folder,
        unzip_file,
    )


@app.cell
def __(np, os, pydicom, series, study, zarr):
    def load_dicom_series(directory):
        dicoms = [
            pydicom.dcmread(os.path.join(directory, f))
            for f in os.listdir(directory)
            if f.endswith(".ima") or f.endswith(".dcm")
        ]
        dicoms.sort(key=lambda x: int(x.InstanceNumber))  # Sort DICOMs by their instance number
        return dicoms


    def dicom_series_to_pixels(series):
        try:
            frames = []
            for ds in series:
                frames.append(np.expand_dims(ds.pixel_array, (0)))
            s = np.vstack(frames)

            return s.astype(np.float32), None
        except Exception as e:
            print("Error in dicom_series_to_pixels:", e)
            return None, str(e)


    def get_dicom_slice_thickness(series):
        try:
            slice_thickness = list(set([ds.SliceThickness for ds in series]))
            if len(slice_thickness) > 1:
                print("DICOMs have different slice thicknesses", slice_thickness)
                raise ValueError("DICOMs have different slice thicknesses")
            return slice_thickness[0], None
        except ValueError as ve:
            return slice_thickness, str(ve)
        except Exception as e:
            return None, str(e)


    def get_dicom_z_spacing(series):
        try:
            spacing_between_slices = list(set([ds.SpacingBetweenSlices for ds in series]))
            if len(spacing_between_slices) > 1:
                print("DICOMs have different spacing between slices", spacing_between_slices)
                raise ValueError("DICOMs have different spacing between slices")
            return spacing_between_slices[0], None
        except ValueError as ve:
            return spacing_between_slices, str(ve)
        except Exception as e:
            return None, str(e)


    def get_dicom_x_spacing(series):
        try:
            x_spacing = list(set([ds.PixelSpacing[0] for ds in series]))
            if len(x_spacing) > 1:
                print("DICOMs have different x spacing", x_spacing)
                raise ValueError("DICOMs have different x spacing")
            return x_spacing[0], None
        except ValueError as ve:
            return x_spacing, str(ve)
        except Exception as e:
            return None, str(e)


    def get_dicom_y_spacing(series):
        try:
            y_spacing = list(set([ds.PixelSpacing[1] for ds in series]))
            if len(y_spacing) > 1:
                print("DICOMs have different y spacing", y_spacing)
                raise ValueError("DICOMs have different y spacing")
            return y_spacing[0], None
        except ValueError as ve:
            return y_spacing, str(ve)
        except Exception as e:
            return None, str(e)


    def get_axis_of_max_variation_and_direction(series):
        try:
            first_position = np.array(series[0].ImagePositionPatient)
            last_position = np.array(series[-1].ImagePositionPatient)

            # Calculate the difference vector
            difference = last_position - first_position

            # Identify the axis of maximum variation
            max_variation_axis = np.argmax(np.abs(difference))
            axis_name = ["SAG", "COR", "AXL"][max_variation_axis]

            # Determine the direction (positive or negative)
            direction = 1 if difference[max_variation_axis] > 0 else -1

            return axis_name, direction, None
        except Exception as e:
            return None, None, str(e)

    def dicom_series_to_image_position_patient(series):
        try:
            frames = []
            for ds in series:
                frames.append(np.expand_dims(ds.ImagePositionPatient, (0)))
            s = np.vstack(frames)

            return s, None
        except Exception as e:
            print("Error in dicom_series_to_image_position_patient:", e)
            return None, str(e)


    def dicom_series_to_image_orientation_patient(series):
        try:
            frames = []
            for ds in series:
                iop = np.array(ds.ImageOrientationPatient).reshape(2, 3)
                frames.append(np.expand_dims(iop, (0)))
            s = np.vstack(frames)

            return s, None
        except Exception as e:
            print("Error in dicom_series_to_image_orientation_patient:", e)
            return None, str(e)


    def pixels_to_zarr_file(arr, zarr_file_path):
        try:
            # Create a Zarr array with the specified chunk size
            zarr_array = zarr.open(
                zarr_file_path, mode="w", shape=arr.shape, chunks=(8, 128, 128), dtype=arr.dtype
            )

            # Store the data into the Zarr array
            zarr_array[:] = arr
        except:
            print(f"Error with {study} {series}")


    def iop_to_zarr_group(arr, zarr_group, study, series):
        root = zarr.open(zarr_group, "a")
        st = root.require_group(study)
        sr = st.require_group(series)
        sr.create_dataset("iop", data=arr)


    def ipp_to_zarr_group(arr, zarr_group, study, series):
        root = zarr.open(zarr_group, "a")
        st = root.require_group(study)
        sr = st.require_group(series)
        sr.create_dataset("ipp", data=arr)
    return (
        dicom_series_to_image_orientation_patient,
        dicom_series_to_image_position_patient,
        dicom_series_to_pixels,
        get_axis_of_max_variation_and_direction,
        get_dicom_slice_thickness,
        get_dicom_x_spacing,
        get_dicom_y_spacing,
        get_dicom_z_spacing,
        iop_to_zarr_group,
        ipp_to_zarr_group,
        load_dicom_series,
        pixels_to_zarr_file,
    )


@app.cell
def __(
    dicom_series_to_image_orientation_patient,
    dicom_series_to_image_position_patient,
    dicom_series_to_pixels,
    get_axis_of_max_variation_and_direction,
    get_dicom_slice_thickness,
    get_dicom_x_spacing,
    get_dicom_y_spacing,
    get_dicom_z_spacing,
    iop_to_zarr_group,
    ipp_to_zarr_group,
    load_dicom_series,
    np,
    os,
    pixels_to_zarr_file,
):
    def process_dicom_series(key, base_path, study_name, series_name, output_dir):
        errors = []

        series_dir = os.path.join(base_path, study_name, series_name)
        dcm = load_dicom_series(series_dir)

        arr, err = dicom_series_to_pixels(dcm)
        if err:
            errors.append("DICOM error")
            mean = None
            std = None
            shape = None
        else:
            mean = np.mean(arr)
            std = np.std(arr)
            shape = arr.shape
            pixels_to_zarr_file(arr, f"{output_dir}/dicoms/{study_name}/{series_name}.zarr")

        z_thickness, err = get_dicom_slice_thickness(dcm)
        if err:
            errors.append(f"Z_THICKNESS")
        z_spacing, err = get_dicom_z_spacing(dcm)
        if err:
            errors.append(f"Z_SPACING")
        x_spacing, err = get_dicom_x_spacing(dcm)
        if err:
            errors.append(f"X_SPACING")
        y_spacing, err = get_dicom_y_spacing(dcm)
        if err:
            errors.append(f"Y_SPACING")

        orientation, direction, err = get_axis_of_max_variation_and_direction(dcm)
        if err:
            errors.append(f"IOP_MISSING")
        
        iop, err = dicom_series_to_image_orientation_patient(dcm)
        if err:
            errors.append(f"IOP_MISSING")
        else:
            iop_to_zarr_group(iop, f"{output_dir}/dataset_slice_metadata.zarr", study_name, series_name)

        ipp, err = dicom_series_to_image_position_patient(dcm)
        if err:
            errors.append(f"IPP_MISSING")
        else:
            ipp_to_zarr_group(ipp, f"{output_dir}/dataset_slice_metadata.zarr", study_name, series_name)

        return {
            'source': key,
            'study': study_name,
            'series': series_name,
            'shape': shape,
            'z_thickness': z_thickness,
            'z_spacing': z_spacing,
            'x_spacing': x_spacing,
            'y_spacing': y_spacing,
            'mean': mean,
            'std': std,
            'orientation': orientation,
            'direction': direction,
            'error': errors
        }
    return process_dicom_series,


@app.cell
def __(mo):
    mo.md(
        r"""
        # RSNA 2024 Data (Kaggle)
        ## RSNA 2024 Lumbar Spine Degenerative Classification Challenge

        [https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification)

        Low back pain is the leading cause of disability worldwide, according to the World Health Organization, affecting 619 million people in 2020. Most people experience low back pain at some point in their lives, with the frequency increasing with age. Pain and restricted mobility are often symptoms of spondylosis, a set of degenerative spine conditions including degeneration of intervertebral discs and subsequent narrowing of the spinal canal (spinal stenosis), subarticular recesses, or neural foramen with associated compression or irritations of the nerves in the low back.

        Magnetic resonance imaging (MRI) provides a detailed view of the lumbar spine vertebra, discs and nerves, enabling radiologists to assess the presence and severity of these conditions. Proper diagnosis and grading of these conditions help guide treatment and potential surgery to help alleviate back pain and improve overall health and quality of life for patients.

        RSNA has teamed with the American Society of Neuroradiology (ASNR) to conduct this competition exploring whether artificial intelligence can be used to aid in the detection and classification of degenerative spine conditions using lumbar spine MR images.

        The challenge will focus on the classification of five lumbar spine degenerative conditions: Left Neural Foraminal Narrowing, Right Neural Foraminal Narrowing, Left Subarticular Stenosis, Right Subarticular Stenosis, and Spinal Canal Stenosis. For each imaging study in the dataset, we’ve provided severity scores (Normal/Mild, Moderate, or Severe) for each of the five conditions across the intervertebral disc levels L1/L2, L2/L3, L3/L4, L4/L5, and L5/S1.

        To create the ground truth dataset, the RSNA challenge planning task force collected imaging data sourced from eight sites on five continents. This multi-institutional, expertly curated dataset promises to improve standardized classification of degenerative lumbar spine conditions and enable development of tools to automate accurate and rapid disease classification.
        """
    )
    return


@app.cell
def __(
    RSNA_2024_competition,
    delete_folder,
    kaggle,
    os,
    raw_data_path,
    rename_folder,
    zipfile,
):
    # check if the folder doesn't already exist
    if not os.path.exists(f'{raw_data_path}/{RSNA_2024_competition}'):
        kaggle.api.competition_download_files(RSNA_2024_competition, path=f'{raw_data_path}/{RSNA_2024_competition}', quiet=False, force=True)

        zip_path = os.path.join(f'{raw_data_path}/{RSNA_2024_competition}', RSNA_2024_competition + '.zip')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(f'{raw_data_path}/unzip_{RSNA_2024_competition}')

        delete_folder(f'{raw_data_path}/{RSNA_2024_competition}')
        rename_folder(f'{raw_data_path}/unzip_{RSNA_2024_competition}', f'{raw_data_path}/{RSNA_2024_competition}')
    return zip_path, zip_ref


@app.cell
def __(RSNA_2024_competition, os, pd, process_dicom_series, raw_data_path):
    # check if the folder doesn't already exist
    if not os.path.exists('/cbica/home/gangarav/RSNA_PROCESSED'):
        # iterate through all of the folders in the raw_data_path
        raw_dicom_folder = f'{raw_data_path}/{RSNA_2024_competition}/train_images'
        output_folder = '/cbica/home/gangarav/RSNA_PROCESSED'

        results = []

        for i, RSNA_study in enumerate(os.listdir(raw_dicom_folder)):
            if i % 100 == 0:
                print(i)
            for RSNA_series in os.listdir(os.path.join(raw_dicom_folder, RSNA_study)):
                results.append(
                    process_dicom_series(
                        "RSNA2024",
                        raw_dicom_folder,
                        RSNA_study,
                        RSNA_series,
                        output_folder
                    )
                )

        df = pd.DataFrame(results)
        df.to_pickle('dataframe.pkl')
    return (
        RSNA_series,
        RSNA_study,
        df,
        i,
        output_folder,
        raw_dicom_folder,
        results,
    )


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(r"""# Lumbar Spine Dataset""")
    return


@app.cell
def __(
    LSD_dataset,
    delete_file,
    download_file,
    os,
    raw_data_path,
    rename_folder,
    unzip_file,
):
    if not os.path.exists(f"{raw_data_path}/{LSD_dataset}"):
        download_file("https://data.mendeley.com/public-files/datasets/k57fr854j2/files/eab74360-db27-4ec5-ade3-7b1b8d88e2db/file_downloaded", f"{raw_data_path}/{LSD_dataset}")
        unzip_file(f"{raw_data_path}/{LSD_dataset}", f"{raw_data_path}/{LSD_dataset}_zip")
        delete_file(f"{raw_data_path}/{LSD_dataset}")
        rename_folder(f"{raw_data_path}/{LSD_dataset}_zip", f"{raw_data_path}/{LSD_dataset}")
    return


@app.cell
def __(LSD_dataset, os, pd, process_dicom_series, raw_data_path):
    raw_dicom_folder = f"{raw_data_path}/{LSD_dataset}/01_MRI_Data"
    output_folder = '/cbica/home/gangarav/LSD_PROCESSED'

    results = []

    for i, LSD_patient in enumerate(os.listdir(raw_dicom_folder)):
        if i % 100 == 0:
            print(i)
        for LSD_study in os.listdir(os.path.join(raw_dicom_folder, LSD_patient)):
            for LSD_series in os.listdir(os.path.join(raw_dicom_folder, LSD_patient, LSD_study)):
                results.append(
                    process_dicom_series(
                        "LSD",
                        raw_dicom_folder,
                        f"{LSD_patient}/{LSD_study}",
                        LSD_series,
                        output_folder
                    )
                )

    df = pd.DataFrame(results)
    df.to_pickle('lsd_dataframe.pkl')
    return (
        LSD_patient,
        LSD_series,
        LSD_study,
        df,
        i,
        output_folder,
        raw_dicom_folder,
        results,
    )


@app.cell
def __(df):
    df
    return


@app.cell
def __(pd):
    pd.read_pickle('dataframe.pkl')
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
