import json
import csv
import argparse
import os

def get_json_files(directories):
    json_files = []
    for directory in directories:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
    return json_files

def write_csv(input_files, output_file):
    # Check if the output file already exists
    file_exists = os.path.isfile(output_file)

    # Open the CSV file for appending
    with open(output_file, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write the header only if the file does not exist
        if not file_exists:
            header = [
                'file_name', 'average_bandwidth', 'average_interarrival', 'average_power', 'average_read_latency',
                'act_energy', 'act_stb_energy', 'pre_stb_energy', 'read_energy', 'ref_energy', 'refb_energy', 'sref_energy', 'total_energy', 'write_energy',
                'num_read_cmds', 'num_read_row_hits', 'num_reads_done',
                'num_write_cmds', 'num_write_row_hits', 'num_writes_done',
                'num_ref_cmds', 'num_refb_cmds', 'num_srefe_cmds', 'num_srefx_cmds',
                'all_bank_idle_cycles', 'channel', 'epoch_num', 'hbm_dual_cmds', 'num_act_cmds', 'num_cycles', 'num_ondemand_pres', 'num_pre_cmds', 'num_write_buf_hits', 'rank_active_cycles', 'sref_cycles'
            ]
            writer.writerow(header)
        
        # Write the data for each input file
        for input_file in input_files:
            # Load the JSON data
            with open(input_file) as json_file:
                data = json.load(json_file)
            
            # Write the data
            for key, value in data.items():
                row = [
                    os.path.join(os.path.basename(os.path.dirname(input_file)), os.path.basename(input_file)),  # Add the directory and file name as the first column
                    value.get('average_bandwidth'), value.get('average_interarrival'), value.get('average_power'), value.get('average_read_latency'),
                    value.get('act_energy'), value.get('act_stb_energy'), value.get('pre_stb_energy'), value.get('read_energy'), value.get('ref_energy'), value.get('refb_energy'), value.get('sref_energy'), value.get('total_energy'), value.get('write_energy'),
                    value.get('num_read_cmds'), value.get('num_read_row_hits'), value.get('num_reads_done'),
                    value.get('num_write_cmds'), value.get('num_write_row_hits'), value.get('num_writes_done'),
                    value.get('num_ref_cmds'), value.get('num_refb_cmds'), value.get('num_srefe_cmds'), value.get('num_srefx_cmds'),
                    value.get('all_bank_idle_cycles'), value.get('channel'), value.get('epoch_num'), value.get('hbm_dual_cmds'), value.get('num_act_cmds'), value.get('num_cycles'), value.get('num_ondemand_pres'), value.get('num_pre_cmds'), value.get('num_write_buf_hits'), value.get('rank_active_cycles'), value.get('sref_cycles')
                ]
                writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert JSON to CSV and append to output file.')
    parser.add_argument('-i', '--input', type=str, nargs='+', required=True, help='Input directories containing JSON files')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output CSV file')
    args = parser.parse_args()
    
    input_files = get_json_files(args.input)
    write_csv(input_files, args.output)