"""按分析字段白名单重建最小 ULog，不复制原始元数据或其他消息。"""
import heapq
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

REQUIRED={
    'drone_fusion':('timestamp','position_x','position_y','position_z','velocity_x','velocity_y','velocity_z','yaw'),
    'fusion_gps':('timestamp','yaw'),
    'sensor_gps':('timestamp','veln','vele','veld','lat','lon'),
}
OPTIONAL={'drone_fusion':('is_hori_valid','is_vert_valid','using_fake_fusion')}
TYPES={'uint64_t':'Q','int64_t':'q','uint32_t':'I','int32_t':'i',
       'uint16_t':'H','int16_t':'h','uint8_t':'B','int8_t':'b','bool':'?',
       'float':'f','double':'d'}


def sanitize(source,destination):
    from pyulog import ULog
    source=Path(source).resolve();destination=Path(destination).resolve()
    if source==destination:raise ValueError('输出文件不能覆盖原始 ULog')
    if destination.exists():raise ValueError('输出文件已存在，请选择新的文件名')
    u=ULog(str(source))
    # 与 fusion_reference.load 的按名称选取行为一致，使用最后一个同名实例。
    topics={d.name:d for d in u.data_list}
    datasets=[]
    for name,required in REQUIRED.items():
        if name not in topics:raise ValueError(f'缺少必要主题：{name}')
        d=topics[name]
        missing=[k for k in required if k not in d.data]
        if missing:raise ValueError(f'{name} 缺少字段：{", ".join(missing)}')
        names=required+tuple(k for k in OPTIONAL.get(name,()) if k in d.data)
        kinds={f.field_name:f.type_str for f in d.field_data}
        if any(kinds[k] not in TYPES for k in names):raise ValueError(f'{name} 包含不支持的字段类型')
        if not len(d.data['timestamp']):raise ValueError(f'{name} 没有数据')
        fmt=struct.Struct('<'+''.join(TYPES[kinds[k]] for k in names))
        datasets.append((name,d,names,kinds,fmt))
    destination.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='ulog_minimal_',suffix='.ulg',dir=destination.parent)
    os.close(fd)
    def message(stream,kind,payload):
        stream.write(struct.pack('<HB',len(payload),ord(kind)));stream.write(payload)
    try:
        with open(temp,'wb') as stream:
            start=min(int(d.data['timestamp'][0]) for _,d,_,_,_ in datasets)
            stream.write(b'ULog\x01\x12\x35'+bytes([1])+struct.pack('<Q',start))
            message(stream,'B',bytes(40))
            for msg_id,(name,d,names,kinds,fmt) in enumerate(datasets):
                definition=name+':'+''.join(f'{kinds[k]} {k};' for k in names)
                message(stream,'F',definition.encode('ascii'))
            for msg_id,(name,d,names,kinds,fmt) in enumerate(datasets):
                message(stream,'A',struct.pack('<BH',0,msg_id)+name.encode('ascii'))
            def records(msg_id,item):
                _,d,names,_,fmt=item
                for index in range(len(d.data['timestamp'])):
                    values=[d.data[k][index].item() for k in names]
                    yield int(d.data['timestamp'][index]),msg_id,struct.pack('<H',msg_id)+fmt.pack(*values)
            for _,_,payload in heapq.merge(*(records(i,item) for i,item in enumerate(datasets))):
                message(stream,'D',payload)
        with open(temp,'rb') as validation_stream:
            check=ULog(validation_stream)
        if len(check.data_list)!=len(datasets):raise ValueError('新 ULog 验证失败：主题数量不一致')
        checked={d.name:d for d in check.data_list}
        for name,d,names,_,_ in datasets:
            if set(checked[name].data)!=set(names):raise ValueError('新 ULog 字段白名单验证失败')
            for key in names:
                if not np.array_equal(d.data[key],checked[name].data[key],equal_nan=True):
                    raise ValueError(f'新 ULog 数据验证失败：{name}.{key}')
        # 使用现有真值加载器验证保留字段的航向旋转和有效性筛选可正常执行。
        # 脱敏不涉及速度序列 yaw_diff 估计，沿用 fusion_gps.yaw / drone_fusion.yaw 路径。
        from .fusion_reference import load
        load(temp,require_position=True,estimate_yaw_diff=False)
        os.rename(temp,destination)
    finally:
        if Path(temp).exists():Path(temp).unlink()
    return [(name,len(d.data['timestamp']),list(names)) for name,d,names,_,_ in datasets]
