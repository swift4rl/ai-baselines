SRC_DIR=proto/mlagents_envs/communicator_objects
DST_DIR_C=./com.unity.ml-agents/Runtime/Grpc/CommunicatorObjects
DST_DIR_S=./Sources/environments
PROTO_PATH=proto
    
#rm -rf $DST_DIR_C
#rm -rf $DST_DIR_S/$SWIFT_PACKAGE
#mkdir -p $DST_DIR_C
#mkdir -p $DST_DIR_S/$SWIFT_PACKAGE

protoc --proto_path=proto --csharp_opt=internal_access --csharp_out $DST_DIR_C $SRC_DIR/*.proto
protoc --proto_path=proto --swift_out=$DST_DIR_S $SRC_DIR/*.proto

GRPC=unity_to_external.proto

protoc --proto_path=proto --csharp_out=$DST_DIR_C --grpc_out=internal_access:$DST_DIR_C $SRC_DIR/$GRPC --plugin=protoc-gen-grpc="/usr/local/bin/grpc_csharp_plugin"
protoc $SRC_DIR/$GRPC \
    --proto_path=proto \
    --swift_out=$DST_DIR_S \
    --grpc-swift_out=Client=true,Server=true:$DST_DIR_S \
    --plugin=protoc-gen-grpc-swift="~/Development/protoc/protoc-4.0.0-rc-2-osx-x86_64/bin/protoc-grpc-swift-plugins-1.0.0-alpha.20/bin/protoc-gen-grpc-swift" # remove ~ make this full path !!!


# Surround UnityToExternal.cs file with macro
echo "#if UNITY_EDITOR || UNITY_STANDALONE_WIN || UNITY_STANDALONE_OSX || UNITY_STANDALONE_LINUX
`cat $DST_DIR_C/UnityToExternalGrpc.cs`
#endif" > $DST_DIR_C/UnityToExternalGrpc.cs
