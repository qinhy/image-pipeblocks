
# Standard Library Imports
import json
from multiprocessing import shared_memory
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import uuid

# Third-Party Library Imports
import numpy as np
from pydantic import BaseModel, Field

class CommonIO:
    class Base(BaseModel):       
        auto_del:bool = True
        def model_dump_json_dict(self):
            return json.loads(self.model_dump_json())
        def write(self,data):
            raise ValueError("[CommonIO.Reader]: This is Reader can not write")
        def read(self):
            raise ValueError("[CommonIO.Writer]: This is Writer can not read") 
        def close(self):
            raise ValueError("[CommonIO.Base]: 'close' not implemented")
    class Reader(Base):
        def read(self)->Any:
            raise ValueError("[CommonIO.Reader]: 'read' not implemented")
    class Writer(Base):
        def write(self,data):
            raise ValueError("[CommonIO.Writer]: 'write' not implemented")

class GeneralSharedMemoryIO(CommonIO):
    class Base(CommonIO.Base):
        shm_name: str = Field(..., description="The name of the shared memory segment")
        create: bool = Field(default=False, description="Flag indicating whether to create or attach to shared memory")
        shm_size: int = Field(..., description="The size of the shared memory segment in bytes")

        _shm:shared_memory.SharedMemory = None
        _buffer:memoryview = None

        def build_buffer(self):
            if hasattr(self,'_shm') and self._shm:return
            # Initialize shared memory with the validated size and sanitized name
            self._shm = shared_memory.SharedMemory(name=self.shm_name, create=self.create, size=self.shm_size)
            self._buffer = memoryview(self._shm.buf)  # View into the shared memory buffer
            return self
                
        def close(self):
            """Detach from the shared memory."""
            # Release the memoryview before closing the shared memory
            if hasattr(self,'_buffer') and self._buffer:
                self._buffer.release()
                del self._buffer
            if hasattr(self,'_shm') and self._shm:
                self._shm.close()  # Detach from shared memory
                if 'writer' in self.id.lower():
                    self._shm.unlink()  # Unlink (remove) the shared memory segment after writing
                del self._shm

        def __del__(self):
            self.close()

    class Reader(CommonIO.Reader, Base):
        id: str= Field(default_factory=lambda:f"GeneralSharedMemoryIO.Reader:{uuid.uuid4()}")
        def read(self, size: int = None) -> bytes:
            """Read binary data from shared memory."""
            if size is None or size > self.shm_size:
                size = self.shm_size  # Read the whole buffer by default
            return bytes(self._buffer[:size])  # Convert memoryview to bytes
  
    class Writer(CommonIO.Writer, Base):
        id: str= Field(default_factory=lambda:f"GeneralSharedMemoryIO.Writer:{uuid.uuid4()}")
        def write(self, data: bytes):
            """Write binary data to shared memory."""
            if len(data) > self.shm_size:
                raise ValueError(f"Data size exceeds shared memory size ({len(data)} > {self.shm_size})")
            
            # Write the binary data to shared memory
            self._buffer[:len(data)] = data
        
        # def close(self):
        #     super().close()
        #     if hasattr(self,'_shm') and self._shm:
        #         self._shm.unlink()  # Unlink (remove) the shared memory segment after writing
    
    @staticmethod
    def reader(shm_name: str, shm_size: int):
        return GeneralSharedMemoryIO.Reader(shm_name=shm_name, create=False, shm_size=shm_size).build_buffer()
    
    @staticmethod
    def writer(shm_name: str, shm_size: int):
        return GeneralSharedMemoryIO.Writer(shm_name=shm_name, create=True, shm_size=shm_size).build_buffer()
   
class NumpyUInt8SharedMemoryIO(GeneralSharedMemoryIO):
    class Base(GeneralSharedMemoryIO.Base):
        array_shape: tuple = Field(..., description="Shape of the NumPy array to store in shared memory")
        _dtype: np.dtype = np.uint8
        _shared_array: np.ndarray = None
        def __init__(self, **kwargs):
            kwargs['shm_size'] = np.prod(kwargs['array_shape']) * np.dtype(np.uint8).itemsize
            super().__init__(**kwargs)
            
        def build_buffer(self):
            super().build_buffer()
            self._shared_array = np.ndarray(self.array_shape, dtype=self._dtype, buffer=self._buffer)
            return self

    class Reader(GeneralSharedMemoryIO.Reader, Base):
        id: str= Field(default_factory=lambda:f"NumpyUInt8SharedMemoryIO.Reader:{uuid.uuid4()}")
        def read(self,copy=True) -> np.ndarray:
            return self._shared_array.copy() if copy else self._shared_array
            # binary_data = super().read(size=self.shm_size)
            # return np.frombuffer(binary_data, dtype=self._dtype).reshape(self.array_shape)
    
    class Writer(GeneralSharedMemoryIO.Writer, Base):
        id: str= Field(default_factory=lambda:f"NumpyUInt8SharedMemoryIO.Writer:{uuid.uuid4()}")
        def write(self, data: np.ndarray):
            if data.shape != self.array_shape:
                raise ValueError(f"Data shape {data.shape} does not match expected shape {self.array_shape}.")
            if data.dtype != self._dtype:
                raise ValueError(f"Data type {data.dtype} does not match expected type {self._dtype}.")            
            self._shared_array[:] = data[:]
            return data

    @staticmethod
    def reader(shm_name: str, array_shape: tuple):
        shm_size = np.prod(array_shape) * np.dtype(np.uint8).itemsize
        return NumpyUInt8SharedMemoryIO.Reader(shm_size=shm_size,
                                    shm_name=shm_name, create=False, array_shape=array_shape).build_buffer()
    
    @staticmethod
    def writer(shm_name: str, array_shape: tuple):
        shm_size = np.prod(array_shape) * np.dtype(np.uint8).itemsize
        return NumpyUInt8SharedMemoryIO.Writer(shm_size=shm_size,
                                    shm_name=shm_name, create=True, array_shape=array_shape).build_buffer()


class CommonStreamIO(CommonIO):
    class Base(CommonIO.Base):
        fps:float = 0
        stream_key: str = 'NULL'
        is_close: bool = False
        auto_del: bool = False
        
        def stream_id(self):
            return f'streams:{self.stream_key}'

        def write(self, data, metadata={}):
            raise ValueError("[CommonStreamIO.Reader]: This is Reader can not write")
        
        def read(self):
            raise ValueError("[CommonStreamIO.Writer]: This is Writer can not read") 
        
        def close(self):
            raise ValueError("[StreamWriter]: 'close' not implemented")
        
        def get_steam_info(self)->dict:
            raise ValueError("[StreamWriter]: 'get_steam_info' not implemented")
            
        def set_steam_info(self,data):
            raise ValueError("[StreamWriter]: 'set_steam_info' not implemented")
        
    class StreamReader(CommonIO.Reader, Base):
        id: str= Field(default_factory=lambda:f"CommonStreamIO.StreamReader:{uuid.uuid4()}")
        
        def read(self)->tuple[Any,dict]:
            return super().read(),{}
        
        def __iter__(self):
            return self

        def __next__(self):
            return self.read()        
        
    class StreamWriter(CommonIO.Writer, Base):
        id: str= Field(default_factory=lambda:f"CommonStreamIO.StreamWriter:{uuid.uuid4()}")

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            tmp = self.model_dump_json_dict()
            tmp['id'] = self.stream_id()
            
        def write(self, data, metadata={}):
            raise ValueError("[StreamWriter]: 'write' not implemented")
        
        def set_steam_info(self,data:dict):
            self.get_controller().update(**data)
            self.get_controller().storage(
                ).set('streams:'+self.stream_key,
                json.loads(json.dumps(data, default=str)))

class GeneralNumpySharedMemoryIO(GeneralSharedMemoryIO):
    class Base(GeneralSharedMemoryIO.Base):
        id: str = Field(default_factory=lambda: f"GeneralNumpySharedMemoryIO:{uuid.uuid4()}")
        array_shape: tuple = Field(..., description="Shape of the NumPy array in shared memory")
        dtype: Literal[            
                'uint8',
                'uint16',
                'uint32',
                'uint64',
                'float16',
                'float32',
                'float64'] = Field(default=np.dtype(np.uint8).name, description="Data type of the NumPy array")
        shm_size:int = 0 # bytes
        _shared_array: np.ndarray = None

        def model_post_init(self, context):
            dts = {
                'uint8':np.uint8,
                'uint16':np.uint16,
                'uint32':np.uint32,
                'uint64':np.uint64,
                'float16':np.float16,
                'float32':np.float32,
                'float64':np.float64,
            }
            self.shm_size = np.prod(self.array_shape) * dts[self.dtype].itemsize
            self.id = self.id.replace('GeneralNumpy',f'Numpy{self.dtype.capitalize()}')
            return super().model_post_init(context)

        def build_buffer(self):
            super().build_buffer()
            self._shared_array = np.ndarray(self.array_shape, dtype=self.dtype, buffer=self._buffer)
            return self

        def get_steam_info(self)->dict:
            return self.model_dump()
        
    class StreamReader(GeneralSharedMemoryIO.Reader, Base):
        id: str = Field(default_factory=lambda: f"GeneralNumpySharedMemoryIO.Reader:{uuid.uuid4()}")
        def read(self, copy=True) -> np.ndarray:
            return self._shared_array.copy() if copy else self._shared_array

    class StreamWriter(GeneralSharedMemoryIO.Writer, Base):
        id: str = Field(default_factory=lambda: f"GeneralNumpySharedMemoryIO.Writer:{uuid.uuid4()}")
        def write(self, data: np.ndarray):
            if data.shape != self.array_shape:
                raise ValueError(f"Data shape {data.shape} does not match expected shape {self.array_shape}.")
            if data.dtype != self.dtype:
                raise ValueError(f"Data dtype {data.dtype} does not match expected dtype {self.dtype}.")
            self._shared_array[:] = data[:]
            return data

    @staticmethod
    def reader(shm_name: str, array_shape: tuple, dtype: np.dtype = np.uint8):
        shm_size = np.prod(array_shape) * np.dtype(dtype).itemsize
        return GeneralNumpySharedMemoryIO.Reader(
            shm_size=shm_size,
            shm_name=shm_name,
            create=False,
            array_shape=array_shape,
            dtype=np.dtype(dtype).name
        ).build_buffer()

    @staticmethod
    def writer(shm_name: str, array_shape: tuple, dtype: np.dtype = np.uint8):
        shm_size = np.prod(array_shape) * np.dtype(dtype).itemsize
        return GeneralNumpySharedMemoryIO.Writer(
            shm_size=shm_size,
            shm_name=shm_name,
            create=True,
            array_shape=array_shape,
            dtype=np.dtype(dtype).name
        ).build_buffer()
    
class NumpyUInt8SharedMemoryStreamIO(GeneralNumpySharedMemoryIO,CommonStreamIO):        
    @staticmethod
    def reader(stream_key: str, array_shape: tuple):
        shm_size = np.prod(array_shape) * np.dtype(np.uint8).itemsize
        shm_name = stream_key.replace(':','_')
        return GeneralNumpySharedMemoryIO.StreamReader(
            shm_name=shm_name, create=False, stream_key=stream_key,dtype='uint8',
            array_shape=array_shape,shm_size=shm_size).build_buffer()
    
    @staticmethod
    def writer(stream_key: str, array_shape: tuple):
        shm_size = np.prod(array_shape) * np.dtype(np.uint8).itemsize
        shm_name = stream_key.replace(':','_')
        return GeneralNumpySharedMemoryIO.StreamWriter(
            shm_name=shm_name, create=True, stream_key=stream_key,dtype='uint8',
            array_shape=array_shape,shm_size=shm_size).build_buffer()

class NumpyFloat32SharedMemoryStreamIO(GeneralNumpySharedMemoryIO,CommonStreamIO):        
    @staticmethod
    def reader(stream_key: str, array_shape: tuple):
        shm_size = np.prod(array_shape) * np.dtype(np.float32).itemsize
        shm_name = stream_key.replace(':','_')
        return GeneralNumpySharedMemoryIO.StreamReader(
            shm_name=shm_name, create=False, stream_key=stream_key,dtype='float32',
            array_shape=array_shape,shm_size=shm_size).build_buffer()
    
    @staticmethod
    def writer(stream_key: str, array_shape: tuple):
        shm_size = np.prod(array_shape) * np.dtype(np.float32).itemsize
        shm_name = stream_key.replace(':','_')
        return GeneralNumpySharedMemoryIO.StreamWriter(
            shm_name=shm_name, create=True, stream_key=stream_key,dtype='float32',
            array_shape=array_shape,shm_size=shm_size).build_buffer()

# if __name__ == '__main__':
#     writer = NumpyUInt8SharedMemoryStreamIO.writer('test:1',(10,10))
#     writer.write(np.ones((10,10),dtype=np.uint8))
