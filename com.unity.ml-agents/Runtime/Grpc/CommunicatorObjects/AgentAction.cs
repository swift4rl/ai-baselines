// <auto-generated>
//     Generated by the protocol buffer compiler.  DO NOT EDIT!
//     source: mlagents_envs/communicator_objects/agent_action.proto
// </auto-generated>
#pragma warning disable 1591, 0612, 3021
#region Designer generated code

using pb = global::Google.Protobuf;
using pbc = global::Google.Protobuf.Collections;
using pbr = global::Google.Protobuf.Reflection;
using scg = global::System.Collections.Generic;
namespace MLAgents.CommunicatorObjects {

  /// <summary>Holder for reflection information generated from mlagents_envs/communicator_objects/agent_action.proto</summary>
  internal static partial class AgentActionReflection {

    #region Descriptor
    /// <summary>File descriptor for mlagents_envs/communicator_objects/agent_action.proto</summary>
    public static pbr::FileDescriptor Descriptor {
      get { return descriptor; }
    }
    private static pbr::FileDescriptor descriptor;

    static AgentActionReflection() {
      byte[] descriptorData = global::System.Convert.FromBase64String(
          string.Concat(
            "CjVtbGFnZW50c19lbnZzL2NvbW11bmljYXRvcl9vYmplY3RzL2FnZW50X2Fj",
            "dGlvbi5wcm90bxIUY29tbXVuaWNhdG9yX29iamVjdHMiSwoQQWdlbnRBY3Rp",
            "b25Qcm90bxIWCg52ZWN0b3JfYWN0aW9ucxgBIAMoAhINCgV2YWx1ZRgEIAEo",
            "AkoECAIQA0oECAMQBEoECAUQBkIfqgIcTUxBZ2VudHMuQ29tbXVuaWNhdG9y",
            "T2JqZWN0c2IGcHJvdG8z"));
      descriptor = pbr::FileDescriptor.FromGeneratedCode(descriptorData,
          new pbr::FileDescriptor[] { },
          new pbr::GeneratedClrTypeInfo(null, null, new pbr::GeneratedClrTypeInfo[] {
            new pbr::GeneratedClrTypeInfo(typeof(global::MLAgents.CommunicatorObjects.AgentActionProto), global::MLAgents.CommunicatorObjects.AgentActionProto.Parser, new[]{ "VectorActions", "Value" }, null, null, null, null)
          }));
    }
    #endregion

  }
  #region Messages
  internal sealed partial class AgentActionProto : pb::IMessage<AgentActionProto>
  #if !GOOGLE_PROTOBUF_REFSTRUCT_COMPATIBILITY_MODE
      , pb::IBufferMessage
  #endif
  {
    private static readonly pb::MessageParser<AgentActionProto> _parser = new pb::MessageParser<AgentActionProto>(() => new AgentActionProto());
    private pb::UnknownFieldSet _unknownFields;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pb::MessageParser<AgentActionProto> Parser { get { return _parser; } }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pbr::MessageDescriptor Descriptor {
      get { return global::MLAgents.CommunicatorObjects.AgentActionReflection.Descriptor.MessageTypes[0]; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    pbr::MessageDescriptor pb::IMessage.Descriptor {
      get { return Descriptor; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public AgentActionProto() {
      OnConstruction();
    }

    partial void OnConstruction();

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public AgentActionProto(AgentActionProto other) : this() {
      vectorActions_ = other.vectorActions_.Clone();
      value_ = other.value_;
      _unknownFields = pb::UnknownFieldSet.Clone(other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public AgentActionProto Clone() {
      return new AgentActionProto(this);
    }

    /// <summary>Field number for the "vector_actions" field.</summary>
    public const int VectorActionsFieldNumber = 1;
    private static readonly pb::FieldCodec<float> _repeated_vectorActions_codec
        = pb::FieldCodec.ForFloat(10);
    private readonly pbc::RepeatedField<float> vectorActions_ = new pbc::RepeatedField<float>();
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public pbc::RepeatedField<float> VectorActions {
      get { return vectorActions_; }
    }

    /// <summary>Field number for the "value" field.</summary>
    public const int ValueFieldNumber = 4;
    private float value_;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public float Value {
      get { return value_; }
      set {
        value_ = value;
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override bool Equals(object other) {
      return Equals(other as AgentActionProto);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool Equals(AgentActionProto other) {
      if (ReferenceEquals(other, null)) {
        return false;
      }
      if (ReferenceEquals(other, this)) {
        return true;
      }
      if(!vectorActions_.Equals(other.vectorActions_)) return false;
      if (!pbc::ProtobufEqualityComparers.BitwiseSingleEqualityComparer.Equals(Value, other.Value)) return false;
      return Equals(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override int GetHashCode() {
      int hash = 1;
      hash ^= vectorActions_.GetHashCode();
      if (Value != 0F) hash ^= pbc::ProtobufEqualityComparers.BitwiseSingleEqualityComparer.GetHashCode(Value);
      if (_unknownFields != null) {
        hash ^= _unknownFields.GetHashCode();
      }
      return hash;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override string ToString() {
      return pb::JsonFormatter.ToDiagnosticString(this);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void WriteTo(pb::CodedOutputStream output) {
    #if !GOOGLE_PROTOBUF_REFSTRUCT_COMPATIBILITY_MODE
      output.WriteRawMessage(this);
    #else
      vectorActions_.WriteTo(output, _repeated_vectorActions_codec);
      if (Value != 0F) {
        output.WriteRawTag(37);
        output.WriteFloat(Value);
      }
      if (_unknownFields != null) {
        _unknownFields.WriteTo(output);
      }
    #endif
    }

    #if !GOOGLE_PROTOBUF_REFSTRUCT_COMPATIBILITY_MODE
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    void pb::IBufferMessage.InternalWriteTo(ref pb::WriteContext output) {
      vectorActions_.WriteTo(ref output, _repeated_vectorActions_codec);
      if (Value != 0F) {
        output.WriteRawTag(37);
        output.WriteFloat(Value);
      }
      if (_unknownFields != null) {
        _unknownFields.WriteTo(ref output);
      }
    }
    #endif

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int CalculateSize() {
      int size = 0;
      size += vectorActions_.CalculateSize(_repeated_vectorActions_codec);
      if (Value != 0F) {
        size += 1 + 4;
      }
      if (_unknownFields != null) {
        size += _unknownFields.CalculateSize();
      }
      return size;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(AgentActionProto other) {
      if (other == null) {
        return;
      }
      vectorActions_.Add(other.vectorActions_);
      if (other.Value != 0F) {
        Value = other.Value;
      }
      _unknownFields = pb::UnknownFieldSet.MergeFrom(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(pb::CodedInputStream input) {
    #if !GOOGLE_PROTOBUF_REFSTRUCT_COMPATIBILITY_MODE
      input.ReadRawMessage(this);
    #else
      uint tag;
      while ((tag = input.ReadTag()) != 0) {
        switch(tag) {
          default:
            _unknownFields = pb::UnknownFieldSet.MergeFieldFrom(_unknownFields, input);
            break;
          case 10:
          case 13: {
            vectorActions_.AddEntriesFrom(input, _repeated_vectorActions_codec);
            break;
          }
          case 37: {
            Value = input.ReadFloat();
            break;
          }
        }
      }
    #endif
    }

    #if !GOOGLE_PROTOBUF_REFSTRUCT_COMPATIBILITY_MODE
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    void pb::IBufferMessage.InternalMergeFrom(ref pb::ParseContext input) {
      uint tag;
      while ((tag = input.ReadTag()) != 0) {
        switch(tag) {
          default:
            _unknownFields = pb::UnknownFieldSet.MergeFieldFrom(_unknownFields, ref input);
            break;
          case 10:
          case 13: {
            vectorActions_.AddEntriesFrom(ref input, _repeated_vectorActions_codec);
            break;
          }
          case 37: {
            Value = input.ReadFloat();
            break;
          }
        }
      }
    }
    #endif

  }

  #endregion

}

#endregion Designer generated code
