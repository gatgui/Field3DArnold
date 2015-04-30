#include <ai.h>
#include <cstdlib>
#include <string>
#include <iostream>
#include <Field3D/InitIO.h>
#include <Field3D/FieldIO.h>
#include <Field3D/Field3DFile.h>
#include <Field3D/FieldInterp.h>
#include <Field3D/DenseField.h>
#include <Field3D/SparseField.h>
#include <Field3D/EmptyField.h>
#include <Field3D/FieldSampler.h>
#include <Field3D/FieldMapping.h>
#include <Field3D/FieldMetadata.h>
#include <OpenEXR/ImathBoxAlgo.h>

// ---

enum FieldDataType
{
   FDT_half = 0,
   FDT_float,
   FDT_double,
   FDT_unknown
};

enum FieldType
{
   FT_dense = 0,
   FT_sparse,
   FT_mac,
   FT_unknown
};

enum SampleMergeType
{
   SMT_add = 0,
   SMT_max,
   SMT_min,
   SMT_avg
};

template <typename ValueType> struct ArnoldType { enum { Value = AI_TYPE_UNDEFINED }; };
template <> struct ArnoldType<Field3D::half> { enum { Value = AI_TYPE_FLOAT }; };
template <> struct ArnoldType<float> { enum { Value = AI_TYPE_FLOAT }; };
template <> struct ArnoldType<double> { enum { Value = AI_TYPE_FLOAT }; };
template <> struct ArnoldType<Field3D::V3h> { enum { Value = AI_TYPE_VECTOR }; };
template <> struct ArnoldType<Field3D::V3f> { enum { Value = AI_TYPE_VECTOR }; };
template <> struct ArnoldType<Field3D::V3d> { enum { Value = AI_TYPE_VECTOR }; };

template <typename ValueType, int ArnoldType>
struct ArnoldValue
{
   static bool Set(const ValueType &, bool, SampleMergeType, AtParamValue *)
   {
      return false;
   }
};

template <typename ValueType>
struct ArnoldValue<ValueType, AI_TYPE_FLOAT>
{
   static bool Set(const ValueType &val, bool merge, SampleMergeType mergeType, AtParamValue *outValue)
   {
      if (merge)
      {
         switch (mergeType)
         {
         case SMT_max:
            outValue->FLT = std::max(outValue->FLT, float(val));
            break;
         case SMT_min:
            outValue->FLT = std::min(outValue->FLT, float(val));
            break;
         case SMT_avg:
         case SMT_add:
         default:
            outValue->FLT += float(val);
         }
      }
      else
      {
         outValue->FLT = float(val);
      }
      
      return true;
   }
};

template <typename DataType>
struct ArnoldValue<FIELD3D_VEC3_T<DataType>, AI_TYPE_VECTOR>
{
   typedef typename FIELD3D_VEC3_T<DataType> ValueType;
   
   static bool Set(const ValueType &val, bool merge, SampleMergeType mergeType, AtParamValue *outValue)
   {
      if (merge)
      {
         switch (mergeType)
         {
         case SMT_max:
            outValue->VEC.x = std::max(outValue->VEC.x, float(val.x));
            outValue->VEC.y = std::max(outValue->VEC.y, float(val.y));
            outValue->VEC.z = std::max(outValue->VEC.z, float(val.z));
            break;
         case SMT_min:
            outValue->VEC.x = std::min(outValue->VEC.x, float(val.x));
            outValue->VEC.y = std::min(outValue->VEC.y, float(val.y));
            outValue->VEC.z = std::min(outValue->VEC.z, float(val.z));
            break;
         case SMT_avg:
         case SMT_add:
         default:
            outValue->VEC.x += float(val.x);
            outValue->VEC.y += float(val.y);
            outValue->VEC.z += float(val.z);
         }
      }
      else
      {
         outValue->VEC.x = float(val.x);
         outValue->VEC.y = float(val.y);
         outValue->VEC.z = float(val.z);
      }
      
      return true;
   }
};


template <typename FieldType>
struct SampleField
{
   typedef typename FieldType::value_type ValueType;
   
   static bool Sample(typename FieldType::Ptr field, const Field3D::V3d &P,
                      int interp, SampleMergeType mergeType,
                      AtParamValue *outValue, AtByte *outType)
   {
      bool merge = (*outType != AI_TYPE_UNDEFINED);
      
      if (merge && ArnoldType<ValueType>::Value != *outType)
      {
         return false;
      }
      
      ValueType val;
      
      switch (interp)
      {
      case AI_VOLUME_INTERP_TRILINEAR:
         {
            typename FieldType::LinearInterp interpolator;
            val = interpolator.sample(*field, P);
         }
         break;
      case AI_VOLUME_INTERP_TRICUBIC:
         {
            typename FieldType::CubicInterp interpolator;
            val = interpolator.sample(*field, P);
         }
         break;
      case AI_VOLUME_INTERP_CLOSEST:
      default:
         {
            Field3D::V3d Pc(std::max(0.5, P.x) - 0.5,
                            std::max(0.5, P.y) - 0.5,
                            std::max(0.5, P.z) - 0.5);
            
            int vx = int(floor(Pc.x));
            int vy = int(floor(Pc.y));
            int vz = int(floor(Pc.z));
            
            val = field->fastValue(vx, vy, vz);
         }
      }
      
      if (ArnoldValue<ValueType, ArnoldType<ValueType>::Value>::Set(val, merge, mergeType, outValue))
      {
         *outType = ArnoldType<ValueType>::Value;
         return true;
      }
      else
      {
         return false;
      }
   }
};

template <typename DataType>
struct SampleField<Field3D::MACField<FIELD3D_VEC3_T<DataType> > >
{
   typedef FIELD3D_VEC3_T<DataType> ValueType;
   typedef Field3D::MACField<ValueType> FieldType;
   
   static bool Sample(typename FieldType::Ptr field, const Field3D::V3d &P,
                      int interp, SampleMergeType mergeType,
                      AtParamValue *outValue, AtByte *outType)
   {
      bool merge = (*outType != AI_TYPE_UNDEFINED);
      
      if (merge && ArnoldType<ValueType>::Value != *outType)
      {
         return false;
      }
      
      FIELD3D_VEC3_T<DataType> val;
      
      switch (interp)
      {
      case AI_VOLUME_INTERP_TRILINEAR:
         {
            typename FieldType::LinearInterp interpolator;
            val = interpolator.sample(*field, P);
         }
         break;
      case AI_VOLUME_INTERP_TRICUBIC:
         {
            typename FieldType::CubicInterp interpolator;
            val = interpolator.sample(*field, P);
         }
         break;
      case AI_VOLUME_INTERP_CLOSEST:
      default:
         {
            Field3D::V3d Pc(std::max(0.5, P.x) - 0.5,
                            std::max(0.5, P.y) - 0.5,
                            std::max(0.5, P.z) - 0.5);
            
            int vx = int(floor(Pc.x));
            int vy = int(floor(Pc.y));
            int vz = int(floor(Pc.z));
            
            val.x = field->uCenter(vx, vy, vz);
            val.y = field->vCenter(vx, vy, vz);
            val.z = field->wCenter(vx, vy, vz);
         }
      }
      
      if (ArnoldValue<ValueType, ArnoldType<ValueType>::Value>::Set(val, merge, mergeType, outValue))
      {
         *outType = ArnoldType<ValueType>::Value;
         return true;
      }
      else
      {
         return false;
      }
   }
};


struct ScalarFieldData
{
   Field3D::SparseField<Field3D::half>::Ptr sparseh;
   Field3D::SparseField<float>::Ptr sparsef;
   Field3D::SparseField<double>::Ptr sparsed;
   
   Field3D::DenseField<Field3D::half>::Ptr denseh;
   Field3D::DenseField<float>::Ptr densef;
   Field3D::DenseField<double>::Ptr densed;
};

struct VectorFieldData
{
   Field3D::SparseField<Field3D::V3h>::Ptr sparseh;
   Field3D::SparseField<Field3D::V3f>::Ptr sparsef;
   Field3D::SparseField<Field3D::V3d>::Ptr sparsed;
   
   Field3D::DenseField<Field3D::V3h>::Ptr denseh;
   Field3D::DenseField<Field3D::V3f>::Ptr densef;
   Field3D::DenseField<Field3D::V3d>::Ptr densed;
   
   Field3D::MACField<Field3D::V3h>::Ptr mach;
   Field3D::MACField<Field3D::V3f>::Ptr macf;
   Field3D::MACField<Field3D::V3d>::Ptr macd;
};


struct FieldData
{
   // empty field type doesn't matter much
   Field3D::EmptyField<float>::Ptr proxy;
   
   std::string partition;
   std::string name;
   size_t index;
   
   Field3D::FieldRes::Ptr base;
   
   FieldType type;
   FieldDataType dataType;
   bool isVector;
   
   ScalarFieldData scalar;
   VectorFieldData vector;
   
   
   bool setup(Field3D::Field3DInputFile *file)
   {
      if (!file)
      {
         return false;
      }
      
      bool found = false;
      
      if (isVector)
      {
         if (!found)
         {
            Field3D::Field<Field3D::V3h>::Vec ll = file->readVectorLayers<Field3D::half>(partition, name);
            
            if (!ll.empty() && index < ll.size())
            {
               dataType = FDT_half;
               base = ll[index];
               
               vector.sparseh = Field3D::field_dynamic_cast<Field3D::SparseField<Field3D::V3h> >(base);
               if (!vector.sparseh)
               {
                  vector.denseh = Field3D::field_dynamic_cast<Field3D::DenseField<Field3D::V3h> >(base);
                  if (!vector.denseh)
                  {
                     vector.mach = Field3D::field_dynamic_cast<Field3D::MACField<Field3D::V3h> >(base);
                     if (vector.mach)
                     {
                        type = FT_mac;
                        found = true;
                     }
                  }
                  else
                  {
                     type = FT_dense;
                     found = true;
                  }
               }
               else
               {
                  type = FT_sparse;
                  found = true;
               }
            }
         }
         
         if (!found)
         {
            Field3D::Field<Field3D::V3f>::Vec ll = file->readVectorLayers<float>(partition, name);
            
            if (!ll.empty() && index < ll.size())
            {
               dataType = FDT_float;
               base = ll[index];
               
               vector.sparsef = Field3D::field_dynamic_cast<Field3D::SparseField<Field3D::V3f> >(base);
               if (!vector.sparsef)
               {
                  vector.densef = Field3D::field_dynamic_cast<Field3D::DenseField<Field3D::V3f> >(base);
                  if (!vector.densef)
                  {
                     vector.macf = Field3D::field_dynamic_cast<Field3D::MACField<Field3D::V3f> >(base);
                     if (vector.macf)
                     {
                        type = FT_mac;
                        found = true;
                     }
                  }
                  else
                  {
                     type = FT_dense;
                     found = true;
                  }
               }
               else
               {
                  type = FT_sparse;
                  found = true;
               }
            }
         }
         
         if (!found)
         {
            Field3D::Field<Field3D::V3d>::Vec ll = file->readVectorLayers<double>(partition, name);
            
            if (!ll.empty() && index < ll.size())
            {
               dataType = FDT_double;
               base = ll[index];
               
               vector.sparsed = Field3D::field_dynamic_cast<Field3D::SparseField<Field3D::V3d> >(base);
               if (!vector.sparsed)
               {
                  vector.densed = Field3D::field_dynamic_cast<Field3D::DenseField<Field3D::V3d> >(base);
                  if (!vector.densed)
                  {
                     vector.macd = Field3D::field_dynamic_cast<Field3D::MACField<Field3D::V3d> >(base);
                     if (vector.macd)
                     {
                        type = FT_mac;
                        found = true;
                     }
                  }
                  else
                  {
                     type = FT_dense;
                     found = true;
                  }
               }
               else
               {
                  type = FT_sparse;
                  found = true;
               }
            }
         }
      }
      else
      {
         if (!found)
         {
            Field3D::Field<Field3D::half>::Vec ll = file->readScalarLayers<Field3D::half>(partition, name);
            
            if (!ll.empty() && index < ll.size())
            {
               dataType = FDT_half;
               base = ll[index];
               
               scalar.sparseh = Field3D::field_dynamic_cast<Field3D::SparseField<Field3D::half> >(base);
               if (!scalar.sparseh)
               {
                  scalar.denseh = Field3D::field_dynamic_cast<Field3D::DenseField<Field3D::half> >(base);
                  if (scalar.denseh)
                  {
                     type = FT_dense;
                     found = true;
                  }
               }
               else
               {
                  type = FT_sparse;
                  found = true;
               }
            }
         }
         
         if (!found)
         {
            Field3D::Field<float>::Vec ll = file->readScalarLayers<float>(partition, name);
            
            if (!ll.empty() && index < ll.size())
            {
               dataType = FDT_float;
               base = ll[index];
               
               scalar.sparsef = Field3D::field_dynamic_cast<Field3D::SparseField<float> >(base);
               if (!scalar.sparsef)
               {
                  scalar.densef = Field3D::field_dynamic_cast<Field3D::DenseField<float> >(base);
                  if (scalar.densef)
                  {
                     type = FT_dense;
                     found = true;
                  }
               }
               else
               {
                  type = FT_sparse;
                  found = true;
               }
            }
         }
         
         if (!found)
         {
            Field3D::Field<double>::Vec ll = file->readScalarLayers<double>(partition, name);
            
            if (!ll.empty() && index < ll.size())
            {
               dataType = FDT_double;
               base = ll[index];
               
               scalar.sparsed = Field3D::field_dynamic_cast<Field3D::SparseField<double> >(base);
               if (!scalar.sparsed)
               {
                  scalar.densed = Field3D::field_dynamic_cast<Field3D::DenseField<double> >(base);
                  if (scalar.densed)
                  {
                     type = FT_dense;
                     found = true;
                  }
               }
               else
               {
                  type = FT_sparse;
                  found = true;
               }
            }
         }
      }
      
      if (!found)
      {
         base = 0;
         dataType = FDT_unknown;
         return false;
      }
      else
      {
         return true;
      }
   }
   
   bool isIdentical(const FieldData &rhs) const
   {
      if (partition != rhs.partition)
      {
         return false;
      }
      
      if (name != rhs.name)
      {
         return false;
      }
      
      if (index != rhs.index)
      {
         return false;
      }
      
      if (dataType != rhs.dataType)
      {
         return false;
      }
      
      if (type != rhs.type)
      {
         return false;
      }
      
      if (isVector != rhs.isVector)
      {
         return false;
      }
      
      if (base && rhs.base)
      {
         if (base->extents() != rhs.base->extents())
         {
            return false;
         }
         
         if (base->dataWindow() != rhs.base->dataWindow())
         {
            return false;
         }
         
         if (!base->mapping()->isIdentical(rhs.base->mapping()))
         {
            return false;
         }
      }
      
      return true;
   }
   
   bool sample(const Field3D::V3d P, int interp, SampleMergeType mergeType, AtParamValue *outValue, AtByte *outType)
   {
      bool rv = false;
      
      switch (type)
      {
      case FT_sparse:
         switch (dataType)
         {
         case FDT_half:
            rv = (isVector ? SampleField<Field3D::SparseField<Field3D::V3h> >::Sample(vector.sparseh, P, interp, mergeType, outValue, outType)
                           : SampleField<Field3D::SparseField<Field3D::half> >::Sample(scalar.sparseh, P, interp, mergeType, outValue, outType));
            break;
         case FDT_float:
            rv = (isVector ? SampleField<Field3D::SparseField<Field3D::V3f> >::Sample(vector.sparsef, P, interp, mergeType, outValue, outType)
                           : SampleField<Field3D::SparseField<float> >::Sample(scalar.sparsef, P, interp, mergeType, outValue, outType));
            break;
         case FDT_double:
            rv = (isVector ? SampleField<Field3D::SparseField<Field3D::V3d> >::Sample(vector.sparsed, P, interp, mergeType, outValue, outType)
                           : SampleField<Field3D::SparseField<double> >::Sample(scalar.sparsed, P, interp, mergeType, outValue, outType));
         default:
            break;
         }
         break;
      case FT_dense:
         switch (dataType)
         {
         case FDT_half:
            rv = (isVector ? SampleField<Field3D::DenseField<Field3D::V3h> >::Sample(vector.denseh, P, interp, mergeType, outValue, outType)
                           : SampleField<Field3D::DenseField<Field3D::half> >::Sample(scalar.denseh, P, interp, mergeType, outValue, outType));
            break;
         case FDT_float:
            rv = (isVector ? SampleField<Field3D::DenseField<Field3D::V3f> >::Sample(vector.densef, P, interp, mergeType, outValue, outType)
                           : SampleField<Field3D::DenseField<float> >::Sample(scalar.densef, P, interp, mergeType, outValue, outType));
            break;
         case FDT_double:
            rv = (isVector ? SampleField<Field3D::DenseField<Field3D::V3d> >::Sample(vector.densed, P, interp, mergeType, outValue, outType)
                           : SampleField<Field3D::DenseField<double> >::Sample(scalar.densed, P, interp, mergeType, outValue, outType));
         default:
            break;
         }
         break;
      case FT_mac:
         if (isVector)
         {
            switch (dataType)
            {
            case FDT_half:
               rv = SampleField<Field3D::MACField<Field3D::V3h> >::Sample(vector.mach, P, interp, mergeType, outValue, outType);
               break;
            case FDT_float:
               rv = SampleField<Field3D::MACField<Field3D::V3f> >::Sample(vector.macf, P, interp, mergeType, outValue, outType);
               break;
            case FDT_double:
               rv = SampleField<Field3D::MACField<Field3D::V3d> >::Sample(vector.macd, P, interp, mergeType, outValue, outType);
            default:
               break;
            }
         }
      default:
         break;
      }
      
      return rv;
   }
};

class VolumeData
{
public:
   
   VolumeData()
      : mNode(0)
      , mF3DFile(0)
      , mIgnoreTransform(false)
      , mVerbose(false)
   {
   }
   
   ~VolumeData()
   {
      reset();
   }
   
   void reset()
   {
      mNode = 0;
      mPartition = "";
      mIgnoreTransform = false;
      
      mFields.clear();
      
      if (mF3DFile)
      {
         delete mF3DFile;
         mF3DFile = 0;
      }
   }
   
   bool isIdentical(const VolumeData &rhs) const
   {
      if (mPath != rhs.mPath)
      {
         return false;
      }
      
      // don't check mPartition directly
      
      PartitionFieldMap::const_iterator pit0 = mFields.begin();
      PartitionFieldMap::const_iterator pit1 = rhs.mFields.begin();
      
      while (pit0 != mFields.end())
      {
         if (pit1 == rhs.mFields.end())
         {
            // not the same number of partitions
            return false;
         }
         
         FieldMap::const_iterator fit0 = pit0->second.begin();
         FieldMap::const_iterator fit1 = pit1->second.begin();
         
         while (fit0 != pit0->second.end())
         {
            if (fit1 == pit1->second.end())
            {
               // not the same number of fields in partition
               return false;
            }
            
            if (!fit0->second.isIdentical(fit1->second))
            {
               return false;
            }
            
            ++fit0;
            ++fit1;
         }
         
         if (fit1 != pit1->second.end())
         {
            // not the same number of fields in partition
            return false;
         }
         
         ++pit0;
         ++pit1;
      }
      
      if (pit1 != rhs.mFields.end())
      {
         // not the same number of partitions
         return false;
      }
      
      // mIgnoreTransform doesn't influence the fields to be read
      
      return true;
   }
   
   bool init(const AtNode *node, const char *user_string, bool noSetup=false)
   {
      reset();
      
      float frame = 1.0f;
      
      // figure out frame value from options
      AtNode *opts = AiUniverseGetOptions();
      const AtUserParamEntry *param = AiNodeLookUpUserParameter(opts, "frame");
      if (param && AiUserParamGetCategory(param) == AI_USERDEF_CONSTANT)
      {
         int ptype = AiUserParamGetType(param);
         switch (ptype)
         {
         case AI_TYPE_BYTE:
            frame = float(AiNodeGetByte(opts, "frame"));
            break;
         case AI_TYPE_INT:
            frame = float(AiNodeGetInt(opts, "frame"));
            break;
         case AI_TYPE_UINT:
            frame = float(AiNodeGetUInt(opts, "frame"));
            break;
         case AI_TYPE_FLOAT:
            frame = AiNodeGetFlt(opts, "frame");
            break;
         default:
            break;
         }
      }
      
      size_t p0;
      size_t p1;
      size_t p2;
      size_t p3;
      size_t p4;
      
      std::string paramString = (user_string ? user_string : "");
      
      p0 = 0;
      p1 = paramString.find('-', p0);
      
      while (p1 != std::string::npos)
      {
         p2 = paramString.find_first_of(" \t\n", p1);
         
         if (p2 != std::string::npos)
         {
            std::string flag = paramString.substr(p1, p2 - p1);
            std::string arg;
            
            AiMsgDebug("[f3d] Flag \"%s\"", flag.c_str());
            
            if (flag != "-ignoreXform" && flag != "-verbose")
            {
               p3 = paramString.find_first_not_of(" \t\n", p2);
                  
               if (p3 == std::string::npos)
               {
                  AiMsgError("[f3d] Missing value for %s flag", flag.c_str());
                  reset();
                  return false;
               }
            }
            else
            {
               p3 = p2;
            }
            
            if (flag == "-file" ||
                flag == "-partition" ||
                flag == "-merge")
            {
               bool inQuotes = false;
               
               if (paramString[p3] == '"')
               {
                  inQuotes = true;
                  p4 = paramString.find('"', p3 + 1);
               }
               else if (paramString[p3] == '\'')
               {
                  inQuotes = true;
                  p4 = paramString.find('\'', p3 + 1);
               }
               else
               {
                  p4 = paramString.find_first_of(" \t\n", p3 + 1);
               }
               
               if (p4 == std::string::npos)
               {
                  if (inQuotes)
                  {
                     AiMsgError("[f3d] Missing value for %s flag", flag.c_str());
                     reset();
                     return false;
                  }
                  else
                  {
                     arg = paramString.substr(p3);
                  }
               }
               else
               {
                  int off = (inQuotes ? 1 : 0);
                  
                  arg = paramString.substr(p3 + off, p4 - p3 - off);
               }
               
               AiMsgDebug("[f3d] %s = \"%s\"", flag.substr(1).c_str(), arg.c_str());
               
               if (flag == "-file")
               {
                  mPath = arg;
               }
               else if (flag == "-partition")
               {
                  mPartition = arg;
               }
               else if (flag == "-merge")
               {
                  size_t m0 = 0;
                  size_t m1 = arg.find(',', m0);
                  
                  while (m0 != std::string::npos)
                  {
                     std::string md = arg.substr(m0, (m1 == std::string::npos ? m1 : m1 - m0));
                     
                     size_t m2 = md.find('=');
                     
                     if (m2 != std::string::npos)
                     {
                        std::string channel = md.substr(0, m2);
                        std::string mtype = md.substr(m2 + 1);
                        
                        if (mtype == "avg")
                        {
                           mChannelsMergeType[channel] = SMT_avg;
                           AiMsgDebug("[f3d] Using AVG merge for channel \"%s\"", channel.c_str());
                        }
                        else if (mtype == "add")
                        {
                           mChannelsMergeType[channel] = SMT_add;
                           AiMsgDebug("[f3d] Using ADD merge for channel \"%s\"", channel.c_str());
                        }
                        else if (mtype == "max")
                        {
                           mChannelsMergeType[channel] = SMT_max;
                           AiMsgDebug("[f3d] Using MAX merge for channel \"%s\"", channel.c_str());
                        }
                        else if (mtype == "min")
                        {
                           mChannelsMergeType[channel] = SMT_min;
                           AiMsgDebug("[f3d] Using MIN merge for channel \"%s\"", channel.c_str());
                        }
                     }
                     
                     m0 = (m1 == std::string::npos ? m1 : m1 + 1);
                     m1 = arg.find(',', m0);
                  }
               }
               
               p0 = (p4 != std::string::npos ? p4 + 1 : p4);
            }
            else if (flag == "-frame")
            {
               float farg = 0.0f;
               
               p4 = paramString.find_first_of(" \t\n", p3 + 1);
               
               arg = paramString.substr(p3, (p4 != std::string::npos ? p4 - p3 : std::string::npos));
               
               if (sscanf(arg.c_str(), "%f", &farg) == 1)
               {
                  frame = farg;
               }
               else
               {
                  AiMsgError("[f3d] Invalid value for -frame: %s", arg.c_str());
               }
               
               p0 = (p4 != std::string::npos ? p4 + 1 : p4);
            }
            else if (flag == "-ignoreXform")
            {
               mIgnoreTransform = true;
               
               p0 = p2 + 1;
               
               AiMsgDebug("[f3d] ignoreTransform = true");
            }
            else if (flag == "-verbose")
            {
               mVerbose = true;
               
               p0 = p2 + 1;
               
               AiMsgDebug("[f3d] verbose = true");
            }
            else
            {
               AiMsgError("[f3d] Invalid flag \"%s\"", flag.c_str());
               reset();
               return false;
            }
            
            p1 = paramString.find('-', p0);
         }
         else
         {
            std::string flag = paramString.substr(p1);
            
            if (flag == "-ignoreXform")
            {
               mIgnoreTransform = true;
            }
            else if (flag == "-verbose")
            {
               mVerbose = true;
            }
            else
            {
               AiMsgError("[f3d] Unknown flag \"%s\"", flag.c_str());
               reset();
               return false;
            }
            
            p0 = std::string::npos;
            p1 = p0;
         }
      }
      
      if (p0 != std::string::npos)
      {
         std::string remain = paramString.substr(p0);
         
         if (remain.find_first_not_of(" \t\n") != std::string::npos)
         {
            // No more flag left in string
            AiMsgError("[f3d] Invalid parameter string (cannot parse \"%s\")", remain.c_str());
            reset();
            return false;
         }
      }
      
      // Replace frame in path (if necessary)
      // allow ###, %03d, or yet <frame> token in file path
      int iframe = int(floorf(frame));
      
      p0 = mPath.find_last_of("\\/");
      
      std::string dirname = (p0 != std::string::npos ? mPath.substr(0, p0) : "");
      std::string basename = (p0 != std::string::npos ? mPath.substr(p0 + 1) : mPath);
      
      // Probably won't ever need more than 32 digits for the frame number
      char *tmp = new char[basename.length() + 32];
      bool foundFramePattern = false;
      
      p0 = basename.rfind("<frame");
      if (p0 != std::string::npos)
      {
         // <frame> or <frame:N>
         p2 = basename.find('>', p0);
         
         if (p2 != std::string::npos)
         {
            p1 = basename.rfind(':', p2);
            
            if (p1 != std::string::npos && p1 > p0)
            {
               std::string pads = basename.substr(p1 + 1, p2 - p1 - 1);
               int pad = 0;
               
               if (sscanf(pads.c_str(), "%d", &pad) != 1)
               {
                  AiMsgWarning("[f3d] Invalid <frame> token format: %s. Assume no padding", basename.substr(p0, p2-p1+1).c_str());
                  pad = 0;
               }
               
               if (pad > 1)
               {
                  sprintf(tmp, "%%0%dd", pad);
                  basename = basename.substr(0, p0) + tmp + basename.substr(p2 + 1);
               }
               else
               {
                  basename = basename.substr(0, p0) + "%d" + basename.substr(p2 + 1);
               }
            }
            else
            {
               basename = basename.substr(0, p0) + "%d" + basename.substr(p2 + 1);
            }
            
            foundFramePattern = true;
         }
      }
      
      if (!foundFramePattern)
      {
         p1 = basename.rfind('#');
         
         if (p1 != std::string::npos)
         {
            if (p1 > 0)
            {
               p0 = basename.find_last_not_of("#", p1 - 1);
               
               if (p0 == std::string::npos)
               {
                  sprintf(tmp, "%%0%lud", p1 + 1);
                  basename = tmp + basename.substr(p1 + 1);
               }
               else
               {
                  size_t n = p1 - p0;
                  
                  sprintf(tmp, "%%0%lud", n);
                  basename = basename.substr(0, p0 + 1) + tmp + basename.substr(p1 + 1);
               }
            }
            else
            {
               basename = std::string("%d") + basename.substr(p1 + 1);
            }
            
            foundFramePattern = true;
         }
         else
         {
            // support already in printf format
         }
      }
      
      if (foundFramePattern)
      {
         AiMsgDebug("[f3d] Replaced pattern in basename: %s", basename.c_str());
      }
      
      sprintf(tmp, basename.c_str(), iframe);
         
      if (basename == tmp)
      {
         AiMsgWarning("[f3d] No frame pattern in file name: \"%s\"", basename.c_str());
      }
      else
      {
         basename = tmp;
      }
      
      delete[] tmp;
      
      mPath = dirname;
      if (mPath.length() > 0)
      {
         mPath += "/";
      }
      mPath += basename;
      
      AiMsgDebug("[f3d] Using %s", mPath.c_str());
      
      if (!noSetup)
      {
         return setup();
      }
      else
      {
         return true;
      }
   }
   
   bool setup()
   {
      AiMsgDebug("[f3d] Open file: %s", mPath.c_str());
      
      mF3DFile = new Field3D::Field3DInputFile();
      
      if (!mF3DFile->open(mPath))
      {
         reset();
         return false;
      }
      else
      {
         std::vector<std::string> partitions;
         std::vector<std::string> layers;
         Field3D::EmptyField<float>::Vec fields;
         FieldMap::iterator fit;
         
         if (mPartition.length() > 0)
         {
            partitions.push_back(mPartition);
         }
         else
         {
            mF3DFile->getPartitionNames(partitions);
         }
         
         for (size_t i=0; i<partitions.size(); ++i)
         {
            const std::string &partition = partitions[i];
            
            AiMsgDebug("[f3d] Found partition \"%s\"", partition.c_str());
            
            FieldMap &partitionFields = mFields[partition];
            
            layers.clear();   
            mF3DFile->getScalarLayerNames(layers, partition);
            
            for (size_t j=0; j<layers.size(); ++j)
            {
               const std::string &layer = layers[j];
               
               AiMsgDebug("[f3d] Found scalar layer \"%s\"", layer.c_str());
               
               fields = mF3DFile->readProxyLayer<float>(partition, layer, false);
               
               if (fields.empty())
               {
                  continue;
               }
               else if (fields.size() == 1)
               {
                  fit = partitionFields.find(layer);
                  if (fit == partitionFields.end())
                  {
                     FieldData &pfield = partitionFields[layer];
                     
                     pfield.type = FT_unknown;
                     pfield.dataType = FDT_unknown;
                     pfield.proxy = fields[0];
                     pfield.isVector = false;
                     pfield.partition = partition;
                     pfield.name = layer;
                     pfield.index = 0;
                     
                     AiMsgDebug("[f3d] Add channel %s.%s", partition.c_str(), layer.c_str());
                  }
                  else
                  {
                     AiMsgWarning("[f3d] Scalar channel with name \"%s\" already exists for partition \"%s\"", layer.c_str(), partition.c_str());
                  }
               }
               else
               {
                  char *tmp = new char[layer.length() + 32];
                  
                  for (size_t k=0; k<fields.size(); ++k)
                  {
                     sprintf(tmp, "%s[%lu]", layer.c_str(), k);
                     
                     fit = partitionFields.find(tmp);
                     if (fit == partitionFields.end())
                     {
                        FieldData &pfield = partitionFields[tmp];
                        
                        pfield.type = FT_unknown;
                        pfield.dataType = FDT_unknown;
                        pfield.proxy = fields[k];
                        pfield.isVector = false;
                        pfield.partition = partition;
                        pfield.name = layer;
                        pfield.index = k;
                        
                        AiMsgDebug("[f3d] Add channel %s.%s[%lu]", partition.c_str(), layer.c_str(), k);
                     }
                     else
                     {
                        AiMsgWarning("[f3d] Scalar channel with name \"%s\" already exists for partition \"%s\"", tmp, partition.c_str());
                     }
                  }
                  
                  delete[] tmp;
               }
            }
            
            layers.clear();
            mF3DFile->getVectorLayerNames(layers, partition);
            
            for (size_t j=0; j<layers.size(); ++j)
            {
               const std::string &layer = layers[j];
               
               AiMsgDebug("[f3d] Found vector layer \"%s\"", layer.c_str());
               
               fields = mF3DFile->readProxyLayer<float>(partition, layer, true);
               
               if (fields.empty())
               {
                  continue;
               }
               else if (fields.size() == 1)
               {
                  fit = partitionFields.find(layer);
                  if (fit == partitionFields.end())
                  {
                     FieldData &pfield = partitionFields[layer];
                     
                     pfield.type = FT_unknown;
                     pfield.dataType = FDT_unknown;
                     pfield.proxy = fields[0];
                     pfield.isVector = true;
                     pfield.partition = partition;
                     pfield.name = layer;
                     pfield.index = 0;
                     
                     AiMsgDebug("[f3d] Add channel %s.%s", partition.c_str(), layer.c_str());
                  }
                  else
                  {
                     AiMsgWarning("[f3d] Vector channel with name \"%s\" already exists for partition \"%s\"", layer.c_str(), partition.c_str());
                  }
               }
               else
               {
                  char *tmp = new char[layer.length() + 32];
                  
                  for (size_t k=0; k<fields.size(); ++k)
                  {
                     sprintf(tmp, "%s[%lu]", layer.c_str(), k);
                     
                     fit = partitionFields.find(tmp);
                     if (fit == partitionFields.end())
                     {
                        FieldData &pfield = partitionFields[tmp];
                        
                        pfield.type = FT_unknown;
                        pfield.dataType = FDT_unknown;
                        pfield.proxy = fields[k];
                        pfield.isVector = true;
                        pfield.partition = partition;
                        pfield.name = layer;
                        pfield.index = k;
                        
                        AiMsgDebug("[f3d] Add channel %s.%s[%lu]", partition.c_str(), layer.c_str(), k);
                     }
                     else
                     {
                        AiMsgWarning("[f3d] Vector channel with name \"%s\" already exists for partition \"%s\"", tmp, partition.c_str());
                     }
                  }
                  
                  delete[] tmp;
               }
            }
         }
         
         return true;
      }
   }
   
   bool update(const AtNode *node, const char *paramString)
   {
      // do not reset if using same file and same fields (same partition)
      // ignore transform and verbose are
      VolumeData tmp;
      
      if (tmp.init(node, paramString))
      {
         if (isIdentical(tmp))
         {
            mNode = node;
            mPartition = tmp.mPartition;
            mIgnoreTransform = tmp.mIgnoreTransform;
            mVerbose = tmp.mVerbose;
         }
         else
         {
            std::swap(mNode, tmp.mNode);
            std::swap(mF3DFile, tmp.mF3DFile);
            std::swap(mPath, tmp.mPath);
            std::swap(mPartition, tmp.mPartition);
            std::swap(mIgnoreTransform, tmp.mIgnoreTransform);
            std::swap(mVerbose, tmp.mVerbose);
            std::swap(mFields, tmp.mFields);
         }
         
         return true;
      }
      else
      {
         return false;
      }
   }
   
   void computeBounds(AtBBox &outBox, float &autoStep)
   {
      Field3D::Box3d bbox;
      
      bbox.makeEmpty();
      
      autoStep = 0.0f;
      
      float autoStepNormalize = 0.0;
      
      for (PartitionFieldMap::const_iterator pit = mFields.begin(); pit != mFields.end(); ++pit)
      {
         for (FieldMap::const_iterator fit = pit->second.begin(); fit != pit->second.end(); ++fit)
         {
            if (!fit->second.proxy)
            {
               continue;
            }
            
            Field3D::V3i res = fit->second.proxy->dataResolution();
            
            Field3D::V3d bmin(0.0, 0.0, 0.0);
            Field3D::V3d bmax(1.0, 1.0, 1.0);
            Field3D::V3d lstep(1.0 / double(res.x),
                               1.0 / double(res.y),
                               1.0 / double(res.z));
            Field3D::V3d step;
            Field3D::Box3d b;
            
            if (!mIgnoreTransform)
            {
               fit->second.proxy->mapping()->localToWorld(bmin, b.min);
               fit->second.proxy->mapping()->localToWorld(bmax, b.max);
               
               // Note: b.min is the origin (0, 0, 0) in world space
               //       localToWorld is transforming its input as a point, not a vector
               fit->second.proxy->mapping()->localToWorld(lstep, step);
               
               step.x = fabs(step.x - b.min.x);
               step.y = fabs(step.y - b.min.y);
               step.z = fabs(step.z - b.min.z);
            }
            else
            {
               b.min = bmin;
               b.max = bmax;
               
               step = lstep;
            }
            
            float s = float(step.x);
            
            if (step.y < s)
            {
               s = float(step.y);
            }
            if (step.z < s)
            {
               s = float(step.z);
            }
            
            autoStep += s;
            autoStepNormalize += 1.0f;
            
            bbox.extendBy(b);
         }
      }
      
      if (!bbox.isEmpty())
      {
         outBox.min.x = float(bbox.min.x);
         outBox.min.y = float(bbox.min.y);
         outBox.min.z = float(bbox.min.z);
         outBox.max.x = float(bbox.max.x);
         outBox.max.y = float(bbox.max.y);
         outBox.max.z = float(bbox.max.z);
         
         autoStep /= autoStepNormalize;
      }
      else
      {
         outBox.min = AI_P3_ZERO;
         outBox.max = AI_P3_ZERO;
         
         autoStep = std::numeric_limits<float>::max();
      }
   }
   
   void rayExtents(const AtVolumeIntersectionInfo *info, AtByte tid, float time, const AtPoint *origin, const AtVector *direction, float t0, float t1)
   {
      AiMsgDebug("[f3d] VolumeData::rayExtents");
      
      // Note: time is not used...
      
      typedef std::pair<float, float> Extent;
      typedef std::vector<Extent> Extents;
      
      Field3D::Box3d box;
      
      box.min = Field3D::V3d(0.0, 0.0, 0.0);
      box.max = Field3D::V3d(1.0, 1.0, 1.0);
      
      Field3D::Ray3d wray;
      
      wray.pos = Field3D::V3d(origin->x, origin->y, origin->z);
      wray.dir = Field3D::V3d(direction->x, direction->y, direction->z);
      
      Extents extents;
      
      AiMsgDebug("[f3d]   Origin: (%f, %f, %f)", wray.pos.x, wray.pos.y, wray.pos.z);
      AiMsgDebug("[f3d]   Direction: (%f, %f, %f)", wray.dir.x, wray.dir.y, wray.dir.z);
      AiMsgDebug("[f3d]   Range: %f -> %f", t0, t1);
      
      for (PartitionFieldMap::const_iterator pit = mFields.begin(); pit != mFields.end(); ++pit)
      {
         for (FieldMap::const_iterator fit = pit->second.begin(); fit != pit->second.end(); ++fit)
         {
            if (!fit->second.proxy)
            {
               continue;
            }
            
            AiMsgDebug("[f3d]   Process field %s.%s[%lu]", fit->second.partition.c_str(), fit->second.name.c_str(), fit->second.index);
            
            Extent extent;
            Field3D::Ray3d ray;
            float escale = 1.0f;
            
            extent.first = -std::numeric_limits<float>::max();
            extent.second = std::numeric_limits<float>::max();
            
            if (!mIgnoreTransform)
            {
               Field3D::V3d tip = wray.pos + wray.dir;
               
               fit->second.proxy->mapping()->worldToLocal(wray.pos, ray.pos);
               fit->second.proxy->mapping()->worldToLocal(tip, ray.dir);
               
               ray.dir -= ray.pos;
               
               escale = ray.dir.length();
               
               if (escale < AI_EPSILON)
               {
                  continue;
               }
               
               // normalize ray direction
               
               escale = 1.0f / escale;
               
               ray.dir.x *= escale;
               ray.dir.y *= escale;
               ray.dir.z *= escale;
               
               AiMsgDebug("[f3d]     Local space origin: (%f, %f, %f)", ray.pos.x, ray.pos.y, ray.pos.z);
               AiMsgDebug("[f3d]     Local space direction: (%f, %f, %f)", ray.dir.x, ray.dir.y, ray.dir.z);
               AiMsgDebug("[f3d]     Extents scale: %f", escale);
            }
            else
            {
               ray.pos = wray.pos;
               ray.dir = wray.dir;
            }
            
            AiMsgDebug("[f3d]     Bounding box min: (%f, %f, %f)", box.min.x, box.min.y, box.min.z);
            AiMsgDebug("[f3d]     Bounding box max: (%f, %f, %f)", box.max.x, box.max.y, box.max.z);
            
            Field3D::V3d in, out;
            
            if (!Imath::findEntryAndExitPoints(ray, box, in, out))
            {
               AiMsgDebug("[f3d]     Doesn't intersect");
               continue;
            }
            
            extent.first = escale * (in - ray.pos).length();
            extent.second = escale * (out - ray.pos).length();
            
            AiMsgDebug("[f3d]     Extents: %f -> %f", extent.first, extent.second);
            
            if (extent.first < extent.second)
            {
               extent.first = std::max(extent.first, t0);
               extent.second = std::min(extent.second, t1);
            }
            
            if (extent.first < extent.second)
            {
               if (extents.size() == 0)
               {
                  extents.push_back(extent);
               }
               else
               {
                  Extents::iterator it = extents.begin();
                  
                  for (; it != extents.end(); ++it)
                  {
                     if (extent.second < it->first)
                     {
                        AiMsgDebug("[f3d]     Before first extent segment");
                        // new extent before current one
                        // if we reach here we also know that we don't overlap with the previous interval
                        extents.insert(it, extent);
                        
                        break;
                     }
                     else
                     {
                        if (extent.first > it->second)
                        {
                           // new extent after current one
                           continue;
                        }
                        else
                        {
                           // Do I really need to check for extent.first < it->first
                           // It should only happend for first extent so the merging logic
                           //   is un-necessary (though it doesn't hurt)
                           
                           if (extent.first < it->first)
                           {
                              AiMsgDebug("[f3d]     Update current extent start");
                              
                              it->first = extent.first;
                              
                              Extents::iterator pit = it;
                              
                              while (pit != extents.begin())
                              {
                                 --pit;
                                 
                                 if (pit->second > it->first)
                                 {
                                    AiMsgDebug("[f3d]     Merge with previous extent segment");
                                    
                                    // merge with previous extent
                                    it->first = pit->first;
                                    it->second = std::max(pit->second, it->second);
                                    
                                    it = extents.erase(pit);
                                    
                                    pit = it;
                                 }
                                 else
                                 {
                                    pit = extents.begin();
                                 }
                              }
                           }
                           
                           if (extent.second > it->second)
                           {
                              AiMsgDebug("[f3d]     Update current extent end");
                              
                              it->second = extent.second;
                              
                              Extents::iterator nit = it;
                              ++nit;
                              
                              while (nit != extents.end())
                              {
                                 if (nit->first < it->second)
                                 {
                                    AiMsgDebug("[f3d]     Merge with next extent segment");
                                    
                                    // merge with next extent
                                    nit->first = it->first;
                                    nit->second = std::max(it->second, nit->second);
                                    
                                    it = extents.erase(it);
                                    
                                    nit = it;
                                    ++nit;
                                 }
                                 else
                                 {
                                    nit = extents.end();
                                 }
                              }
                           }
                           else
                           {
                              // new extent completely inside existing one
                              AiMsgDebug("[f3d]     Contained in existing extent segment");
                           }
                           
                           break;
                        }
                     }
                  }
                  
                  if (it == extents.end())
                  {
                     AiMsgDebug("[f3d]     After last extent segment");
                     
                     extents.push_back(extent);
                  }
               }
            }
            else
            {
               AiMsgDebug("[f3d]     Doesn't intersect");
            }
         }
      }
      
      
      AiMsgDebug("[f3d]   Extents in [%f, %f]", t0, t1);
      for (size_t i=0; i<extents.size(); ++i)
      {
         AiMsgDebug("[f3d]     [%lu]: %f -> %f", i, extents[i].first, extents[i].second);
         
         if (info)
         {
            AiVolumeAddIntersection(info, extents[i].first, extents[i].second);
         }
      }
   }
   
   bool sample(const char *channel, const AtShaderGlobals *sg, int interp, AtParamValue *value, AtByte *type)
   {
      AiMsgDebug("[f3d] VolumeData::sample");
      
      if (!sg || !value || !type)
      {
         return false;
      }
      
      size_t cmplen = strlen(channel);
      
      Field3D::Box3d unitCube;
      
      unitCube.min = Field3D::V3d(0.0, 0.0, 0.0);
      unitCube.max = Field3D::V3d(1.0, 1.0, 1.0);
      
      int totalCount = 0;
      int hitCount = 0;
      
      std::map<std::string, SampleMergeType>::const_iterator mtit = mChannelsMergeType.find(channel);
      SampleMergeType mergeType = (mtit != mChannelsMergeType.end() ? mtit->second : SMT_add);
      
      *type = AI_TYPE_UNDEFINED;
      
      for (PartitionFieldMap::iterator pit = mFields.begin(); pit != mFields.end(); ++pit)
      {
         for (FieldMap::iterator fit = pit->second.begin(); fit != pit->second.end(); ++fit)
         {
            if (!strncmp(channel, fit->first.c_str(), cmplen))
            {
               AiMsgDebug("[f3d]   Process field %s.%s[%lu]", fit->second.partition.c_str(), fit->second.name.c_str(), fit->second.index);
               
               ++totalCount;
               
               // check if inside nox
               Field3D::V3d Pw(sg->Po.x, sg->Po.y, sg->Po.z);
               Field3D::V3d Pl;
               Field3D::V3d Pv;
               
               if (mIgnoreTransform)
               {
                  Pl = Pw;
                  fit->second.proxy->mapping()->localToVoxel(Pw, Pv);
               }
               else
               {
                  fit->second.proxy->mapping()->worldToLocal(Pw, Pl);
                  fit->second.proxy->mapping()->worldToVoxel(Pw, Pv);
               }
               
               if (!unitCube.intersects(Pl))
               {
                  AiMsgDebug("[f3d]     Not in volume");
                  // what value to set?
               }
               else
               {
                  if (!fit->second.base && !fit->second.setup(mF3DFile))
                  {
                     AiMsgWarning("[f3d]     Could not setup field");
                     continue;
                  }
                  
                  if (fit->second.sample(Pv, interp, mergeType, value, type))
                  {
                     ++hitCount;
                  }
               }
            }
         }
      }
      
      if (hitCount > 1 && mergeType == SMT_avg)
      {
         AiMsgDebug("[f3d]   Averaging result");
         
         float scl = 1.0f / float(hitCount);
         
         if (*type == AI_TYPE_FLOAT)
         {
            value->FLT *= scl;
         }
         else if (*type == AI_TYPE_VECTOR)
         {
            value->VEC.x *= scl;
            value->VEC.y *= scl;
            value->VEC.z *= scl;
         }
      }
      
      return (hitCount > 0);
   }

private:
   
   typedef std::map<std::string, FieldData> FieldMap;
   typedef std::map<std::string, FieldMap> PartitionFieldMap;
   
   // fill in with whatever necessary
   const AtNode *mNode;
   Field3D::Field3DInputFile *mF3DFile;
   
   // from parameter string or user attributes
   std::string mPath;
   std::string mPartition;
   bool mIgnoreTransform;
   bool mVerbose;
   
   std::map<std::string, SampleMergeType> mChannelsMergeType;
   
   // field contents
   PartitionFieldMap mFields;
};

// ---

#if AI_VERSION_ARCH_NUM > 4
#   define HAS_VOLUME_UPDATE
#elif AI_VERSION_ARCH_NUM == 4
#   if AI_VERSION_MAJOR_NUM > 2
#      define HAS_VOLUME_UPDATE
#   elif AI_VERSION_MAJOR_NUM == 2
#      if AI_VERSION_MINOR_NUM >= 6
#         define HAS_VOLUME_UPDATE
#      endif
#   endif
#endif

bool F3D_Init(void **user_ptr)
{
   Field3D::initIO();
   return true;
}

bool F3D_Cleanup(void *user_ptr)
{
   return true;
}

bool F3D_CreateVolume(void *user_ptr, const char *user_string, const AtNode *node, AtVolumeData *data)
{
   VolumeData *volume_data = new VolumeData();
   
   bool rv = volume_data->init(node, user_string);
   
   if (rv)
   {
      volume_data->computeBounds(data->bbox, data->auto_step_size);
   }
   else
   {
      data->bbox.min = AI_P3_ZERO;
      data->bbox.max = AI_P3_ZERO;
      data->auto_step_size = std::numeric_limits<float>::max();
   }
   
   data->private_info = (void*) volume_data;
   
   return rv;
}

#ifdef HAS_VOLUME_UPDATE
bool F3D_UpdateVolume(void *user_ptr, const char *user_string, const AtNode *node, AtVolumeData *data)
{
   VolumeData *volume_data = (VolumeData*) data->private_info;
   
   if (volume_data)
   {
      if (volume_data->update(node, user_string))
      {
         volume_data->computeBounds(data->bbox, data->auto_step_size);
         return true;
      }
      else
      {
         data->bbox.min = AI_P3_ZERO;
         data->bbox.max = AI_P3_ZERO;
         data->auto_step_size = std::numeric_limits<float>::max();
         return false;
      }
   }
   else
   {
      AiMsgDebug("[f3d] F3D_UpdateVolume: private_info is not set");
      data->bbox.min = AI_P3_ZERO;
      data->bbox.max = AI_P3_ZERO;
      data->auto_step_size = std::numeric_limits<float>::max();
      return false;
   }
}
#endif

bool F3D_CleanupVolume(void *user_ptr, AtVolumeData *data, const AtNode *node)
{
   VolumeData *volume_data = (VolumeData*) data->private_info;
   
   delete volume_data;
   
   return true;
}

void F3D_RayExtents(void *user_ptr, const AtVolumeData *data, const AtVolumeIntersectionInfo *info, AtByte tid, float time, const AtPoint *origin, const AtVector *direction, float t0, float t1)
{
   VolumeData *volume_data = (VolumeData*) data->private_info;
   
   if (volume_data)
   {
      volume_data->rayExtents(info, tid, time, origin, direction, t0, t1);
   }
}

bool F3D_Sample(void *user_ptr, const AtVolumeData *data, const char *channel, const AtShaderGlobals *sg, int interp, AtParamValue *value, AtByte *type)
{
   VolumeData *volume_data = (VolumeData*) data->private_info;
   
   if (volume_data)
   {
      return volume_data->sample(channel, sg, interp, value, type);
   }
   else
   {
      return false;
   }
}

volume_plugin_loader
{
   vtable->Init = F3D_Init;
   vtable->Cleanup = F3D_Cleanup;
   vtable->CreateVolume = F3D_CreateVolume;
   #ifdef HAS_VOLUME_UPDATE
   vtable->UpdateVolume = F3D_UpdateVolume;
   #endif
   vtable->CleanupVolume = F3D_CleanupVolume;
   vtable->Sample = F3D_Sample;
   vtable->RayExtents = F3D_RayExtents;
   strcpy(vtable->version, AI_VERSION);
   return true;
}

#ifdef ARNOLD_F3D_TEST

int main(int argc, char **argv)
{
   std::string args = "";
   
   for (int i=1; i<argc; ++i)
   {
      args += argv[i];
      if (i+1 < argc)
      {
         args += " ";
      }
   }
   
   AiBegin();
   AiMsgSetConsoleFlags(AI_LOG_ALL);
   
   void *user_ptr = 0;
   
   if (!F3D_Init(&user_ptr))
   {
      AiMsgError("[f3d] F3D_Init failed");
   }
   else
   {
      AtVolumeData data;
      AtNode *node = 0;
      float t = 1.0f;
      AtByte tid = 0;
      AtPoint origin = {0.0f, 0.0f, 2.0f};
      AtVector direction = {0.0f, 0.0f, -1.0f};
      float t0 = 0.0f;
      float t1 = 10.0f;
      
      if (F3D_CreateVolume(user_ptr, args.c_str(), node, &data))
      {
         AiMsgInfo("[f3d] Auto step size = %f", data.auto_step_size);
         
         AiMsgInfo("[f3d] Bounding box min = (%f, %f, %f)", data.bbox.min.x, data.bbox.min.y, data.bbox.min.z);
         AiMsgInfo("[f3d] Bounding box max = (%f, %f, %f)", data.bbox.max.x, data.bbox.max.y, data.bbox.max.z);
         
         F3D_RayExtents(user_ptr, &data, (const AtVolumeIntersectionInfo*)0, tid, t, &origin, &direction, t0, t1);
         
         // Sample
         AtByte outType = AI_TYPE_UNDEFINED;
         AtParamValue outValue;
         
         AtShaderGlobals *sg = AiShaderGlobals();
         sg->Po.x = 0.0f;
         sg->Po.y = 0.0f;
         sg->Po.z = 0.0f;
         
         F3D_Sample(user_ptr, &data, "density", sg, AI_VOLUME_INTERP_CLOSEST, &outValue, &outType);
         
         if (outType == AI_TYPE_FLOAT)
         {
            AiMsgInfo("[f3d] FLT = %f", outValue.FLT);
         }
         else if (outType == AI_TYPE_VECTOR)
         {
            AiMsgInfo("[f3d] VEC = %f, %f, %f", outValue.VEC.x, outValue.VEC.y, outValue.VEC.z);
         }
         else
         {
            AiMsgInfo("[f3d] Unsupported output type");
         }
         
         AiShaderGlobalsDestroy(sg);
         
         F3D_CleanupVolume(user_ptr, &data, node);
      }
      else
      {
         AiMsgError("[f3d] F3D_CreateVolume failed");
      }
      
      F3D_Cleanup(user_ptr);
   }
   
   AiEnd();
   
   return 0;
}


#endif
