import json

def generate_gff(json_file_path, gff_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    genome_id = data['metadata']['genome_id']
    features = data['features']

    with open(gff_file_path, 'w') as gff_file:
        # 写入GFF文件头部信息
        gff_file.write(f"##gff-version 3\n")
        gff_file.write(f"##sequence-region {genome_id} 1 {data['metadata']['length']}\n")

        for feature in features:
            type_ = feature['type']
            location = feature['location']
            start, end = map(int, location.split('..'))
            strand = feature['strand']
            qualifiers = feature['qualifiers']
            confidence = qualifiers['confidence']
            protein_length = qualifiers['protein_length']

            # 构建GFF文件的一行记录
            attributes = f"ID={type_};confidence={confidence};protein_length={protein_length}"
            gff_line = f"{genome_id}\t.\t{type_}\t{start}\t{end}\t.\t{strand}\t.\t{attributes}\n"
            gff_file.write(gff_line)