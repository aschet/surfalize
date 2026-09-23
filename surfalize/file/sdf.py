from datetime import datetime

import sdfio

from ..exceptions import CorruptedFileError
from .common import FileHandler, RawSurface, get_unit_conversion

FIXED_UNIT = "m"
M_TO_UM = get_unit_conversion(FIXED_UNIT, "um")
UM_TO_M = get_unit_conversion("um", FIXED_UNIT)

@FileHandler.register_reader(suffix=".sdf", magic=sdfio.MAGICS)
def read_sdf(filehandle, read_image_layers=False, encoding="utf-8"):
    try:
        sdf = sdfio.SdfFile.loads(filehandle.read())
    except sdfio.SdfError as error:
        raise CorruptedFileError(str(error)) from error

    header = sdf.header
    # Trailer fields first, so they can't clobber the structural header
    # fields below if a name happens to collide.
    metadata = dict(sdf.trailer_fields)
    metadata.update({
        "ManufacID": header.manufacturer_id,
        "CreateDate": header.create_date,
        "ModDate": header.mod_date,
        "NumPoints": header.num_points,
        "NumProfiles": header.num_profiles,
        "Xscale": header.x_scale,
        "Yscale": header.y_scale,
        "Zscale": header.z_scale,
        "Zresolution": header.z_resolution,
        "Compression": 0,
        "DataType": int(header.data_type),
        "CheckType": 0,
        "Dialect": str(header.dialect),
    })
    if header.create_date is not None:
        metadata["timestamp"] = header.create_date

    data = sdf.data * M_TO_UM
    step_x = header.x_scale * M_TO_UM
    step_y = header.y_scale * M_TO_UM
    return RawSurface(data, step_x, step_y, metadata=metadata, image_layers=None)


@FileHandler.register_writer(suffix=".sdf")
def write_sdf(filehandle, surface, encoding="utf-8", binary=True):
    mod_date = datetime.now()
    create_date = surface.metadata.get("timestamp", mod_date)

    data = surface.data.astype("float64") * UM_TO_M
    x_scale = surface.step_x * UM_TO_M
    y_scale = surface.step_y * UM_TO_M

    metadata = sdfio.SdfMetadata(
        dialect=sdfio.SdfDialect.ISO_1_0,
        manufacturer_id="surfalize",
        create_date=create_date,
        mod_date=mod_date,
        data_type=sdfio.DataType.BINARY64,
    )
    file_format = sdfio.FileFormat.BINARY if binary else sdfio.FileFormat.ASCII
    trailer = "" if binary else sdfio.format_tagged_fields({"ExportedBy": "Surfalize"})
    sdfio.write(
        filehandle,
        data,
        x_scale=x_scale,
        y_scale=y_scale,
        z_scale=1.0,
        format=file_format,
        metadata=metadata,
        trailer=trailer,
    )
