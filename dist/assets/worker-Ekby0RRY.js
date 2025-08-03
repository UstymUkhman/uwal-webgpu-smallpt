function r0(s,e){return e.forEach(function(t){t&&typeof t!="string"&&!Array.isArray(t)&&Object.keys(t).forEach(function(n){if(n!=="default"&&!(n in s)){var r=Object.getOwnPropertyDescriptor(t,n);Object.defineProperty(s,n,r.get?r:{enumerable:!0,get:function(){return t[n]}})}})}),Object.freeze(s)}class i0{dims=[];paddedDims=[];layout="x";dataType="Float32";getByteSize(){let e=1;for(const t of this.paddedDims)e*=t;return this.dataType==="Float32"?e*=4:this.dataType==="Float16"&&(e*=2),e}}class o0{desc;data;constructor(e,t){this.desc=e,this.data=t}}class a0{_view;offset=0;constructor(e){this._view=e}read(e){const t=this._view,n=this.offset;switch(this.offset+=e,e){case 1:return t.getUint8(n);case 2:return t.getUint16(n,!0);case 4:return t.getUint32(n,!0);case 8:return Number(t.getBigUint64(n,!0));default:throw new Error("unsupported read size")}}}function l0(s){const e=new Uint8Array(s),t=new a0(new DataView(s));if(t.read(2)!==16855)throw new Error("invalid or corrupted weights blob");const r=t.read(1);if(t.read(1),r!==2)throw new Error("unsupported weights blob version");const i=t.read(8);t.offset=i;const o=t.read(4),a=new Map;for(let l=0;l<o;++l){const u=new i0,c=t.read(2),h=new TextDecoder().decode(e.subarray(t.offset,t.offset+c));t.offset+=c;const d=t.read(1);for(let S=0;S<d;++S)u.dims.push(t.read(4));u.paddedDims=[...u.dims],new TextDecoder().decode(e.subarray(t.offset,t.offset+d))==="oihw"&&(u.layout="oihw"),t.offset+=d;const k=String.fromCharCode(t.read(1));if(k==="f")u.dataType="Float32";else if(k==="h")u.dataType="Float16";else throw new Error("invalid tensor data type");const A=t.read(8),m=e.slice(A,A+u.getByteSize());a.set(h,new o0(u,m))}return a}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const u0=1e-7,c0=1e-4;class h0{constructor(e,t){this.backend=e,this.dataMover=t,this.data=new WeakMap,this.dataIdsCount=0}get(e){return this.data.has(e)||this.dataMover.moveData(this.backend,e),this.data.get(e)}set(e,t){this.dataIdsCount++,this.data.set(e,t)}has(e){return this.data.has(e)}delete(e){return this.dataIdsCount--,this.data.delete(e)}numDataIds(){return this.dataIdsCount}}class jd{refCount(e){return Ot("refCount")}incRef(e){return Ot("incRef")}timerAvailable(){return!0}time(e){return Ot("time")}read(e){return Ot("read")}readSync(e){return Ot("readSync")}readToGPU(e,t){return Ot("readToGPU")}numDataIds(){return Ot("numDataIds")}disposeData(e,t){return Ot("disposeData")}write(e,t,n){return Ot("write")}move(e,t,n,r,i){return Ot("move")}createTensorFromGPUData(e,t,n){return Ot("createTensorFromGPUData")}memory(){return Ot("memory")}floatPrecision(){return Ot("floatPrecision")}epsilon(){return this.floatPrecision()===32?u0:c0}dispose(){return Ot("dispose")}}function Ot(s){throw new Error(`'${s}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function f0(s){let e=s.length,t=0;for(;e>0;)t=Math.random()*e|0,e--,Pr(s,e,t)}function Pr(s,e,t){const n=s[e];s[e]=s[t],s[t]=n}function d0(s){let e=0;for(let t=0;t<s.length;t++)e+=s[t];return e}function R(s,e){if(!s)throw new Error(typeof e=="string"?e:e())}function p0(s,e,t=""){R(cn(s,e),()=>t+` Shapes ${s} and ${e} must match`)}function Kd(s){R(s!=null,()=>"The input to the tensor constructor must be a non-null value.")}function me(s){if(s.length===0)return 1;let e=s[0];for(let t=1;t<s.length;t++)e*=s[t];return e}function cn(s,e){if(s===e)return!0;if(s==null||e==null||s.length!==e.length)return!1;for(let t=0;t<s.length;t++)if(s[t]!==e[t])return!1;return!0}function Su(s){return s%1===0}function da(s,e){return e<=s.length?s:s+" ".repeat(e-s.length)}function m0(s,e){let t=1,n=-1;for(let i=0;i<s.length;++i)if(s[i]>=0)t*=s[i];else if(s[i]===-1){if(n!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${n} and dim ${i}`);n=i}else if(s[i]<0)throw Error(`Shapes can not be < 0. Found ${s[i]} at dim ${i}`);if(n===-1){if(e>0&&e!==t)throw Error(`Size(${e}) must match the product of shape ${s}`);return s}if(t===0)throw Error(`Cannot infer the missing size in [${s}] when there are 0 elements`);if(e%t!==0)throw Error(`The implicit shape can't be a fractional number. Got ${e} / ${t}`);const r=s.slice();return r[n]=e/t,r}function Co(s,e){const t=e.length;return s=s==null?e.map((n,r)=>r):[].concat(s),R(s.every(n=>n>=-t&&n<t),()=>`All values in axis param must be in range [-${t}, ${t}) but got axis ${s}`),R(s.every(n=>Su(n)),()=>`All values in axis param must be integers but got axis ${s}`),s.map(n=>n<0?t+n:n)}function g0(s,e){const t=[],n=[],r=e!=null&&Array.isArray(e)&&e.length===0,i=e==null||r?null:Co(e,s).sort();let o=0;for(let a=0;a<s.length;++a){if(i!=null){if(i[o]===a&&s[a]!==1)throw new Error(`Can't squeeze axis ${a} since its dim '${s[a]}' is not 1`);(i[o]==null||i[o]>a)&&s[a]===1&&(t.push(s[a]),n.push(a)),i[o]<=a&&o++}s[a]!==1&&(t.push(s[a]),n.push(a))}return{newShape:t,keptDims:n}}function li(s,e){return nt(s,e)}function nt(s,e){let t=null;if(s==null||s==="float32")t=new Float32Array(e);else if(s==="int32")t=new Int32Array(e);else if(s==="bool")t=new Uint8Array(e);else if(s==="string")t=new Array(e);else throw new Error(`Unknown data type ${s}`);return t}function y0(s,e){for(let t=0;t<s.length;t++){const n=s[t];if(isNaN(n)||!isFinite(n))throw Error(`A tensor of type ${e} being uploaded contains ${n}.`)}}function b0(s){return s==="bool"||s==="complex64"||s==="float32"||s==="int32"||s==="string"}function ku(s){if(s==="float32"||s==="int32")return 4;if(s==="complex64")return 8;if(s==="bool")return 1;throw new Error(`Unknown dtype ${s}`)}function w0(s){if(s==null)return 0;let e=0;return s.forEach(t=>e+=t.length),e}function Tl(s){return typeof s=="string"||s instanceof String}function x0(s){return typeof s=="boolean"}function Iu(s){return typeof s=="number"}function No(s){return Array.isArray(s)?No(s[0]):s instanceof Float32Array?"float32":s instanceof Int32Array||s instanceof Uint8Array||s instanceof Uint8ClampedArray?"int32":Iu(s)?"float32":Tl(s)?"string":x0(s)?"bool":"float32"}function Eu(s){return!!(s&&s.constructor&&s.call&&s.apply)}function Zt(s){const e=s.length;if(e<2)return[];const t=new Array(e-1);t[e-2]=s[e-1];for(let n=e-3;n>=0;--n)t[n]=t[n+1]*s[n+1];return t}function Xd(s,e,t,n=!1){const r=new Array;if(e.length===1){const i=e[0]*(n?2:1);for(let o=0;o<i;o++)r[o]=t[s+o]}else{const i=e[0],o=e.slice(1),a=o.reduce((l,u)=>l*u)*(n?2:1);for(let l=0;l<i;l++)r[l]=Xd(s+l*a,o,t,n)}return r}function Mh(s,e,t=!1){if(s.length===0)return e[0];const n=s.reduce((r,i)=>r*i)*(t?2:1);if(n===0)return[];if(n!==e.length)throw new Error(`[${s}] does not match the input size ${e.length}${t?" for a complex tensor":""}.`);return Xd(0,s,e,t)}function Kl(s,e){if(Array.isArray(s))return s;if(e==="float32")return s instanceof Float32Array?s:new Float32Array(s);if(e==="int32")return s instanceof Int32Array?s:new Int32Array(s);if(e==="bool"||e==="string")return Uint8Array.from(new Int32Array(s));throw new Error(`Unknown dtype ${e}`)}function Yd(s,e){const t=Rs(s,e);for(let n=0;n<t.length;n++)t[n]=1;return t}function Rs(s,e){if(e==null||e==="float32"||e==="complex64")return new Float32Array(s);if(e==="int32")return new Int32Array(s);if(e==="bool")return new Uint8Array(s);throw new Error(`Unknown data type ${e}`)}function bs(s){s.forEach(e=>{R(Number.isInteger(e)&&e>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${s}].`)})}function Tu(s,e,t){if(e===0)return 0;if(e===1)return s[0];let n=s[s.length-1];for(let r=0;r<s.length-1;++r)n+=t[r]*s[r];return n}function gc(s,e,t){if(e===0)return[];if(e===1)return[s];const n=new Array(e);for(let r=0;r<n.length-1;++r)n[r]=Math.floor(s/t[r]),s-=n[r]*t[r];return n[n.length-1]=s,n}function yc(s){return s&&s.then&&typeof s.then=="function"}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ph="tfjsflags";class _0{constructor(e){this.global=e,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=v0,this.populateURLFlags()}setPlatform(e,t){this.platform!=null&&(ge().getBool("IS_TEST")||ge().getBool("PROD")||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${e}.`)),this.platformName=e,this.platform=t}registerFlag(e,t,n){if(this.flagRegistry[e]={evaluationFn:t,setHook:n},this.urlFlags[e]!=null){const r=this.urlFlags[e];ge().getBool("IS_TEST")||ge().getBool("PROD")||console.warn(`Setting feature override from URL ${e}: ${r}.`),this.set(e,r)}}async getAsync(e){return e in this.flags?this.flags[e]:(this.flags[e]=await this.evaluateFlag(e),this.flags[e])}get(e){if(e in this.flags)return this.flags[e];const t=this.evaluateFlag(e);if(yc(t))throw new Error(`Flag ${e} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[e]=t,this.flags[e]}getNumber(e){return this.get(e)}getBool(e){return this.get(e)}getString(e){return this.get(e)}getFlags(){return this.flags}get features(){return this.flags}set(e,t){if(this.flagRegistry[e]==null)throw new Error(`Cannot set flag ${e} as it has not been registered.`);this.flags[e]=t,this.flagRegistry[e].setHook!=null&&this.flagRegistry[e].setHook(t)}evaluateFlag(e){if(this.flagRegistry[e]==null)throw new Error(`Cannot evaluate flag '${e}': no evaluation function found.`);return this.flagRegistry[e].evaluationFn()}setFlags(e){this.flags=Object.assign({},e)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(typeof this.global>"u"||typeof this.global.location>"u"||typeof this.global.location.search>"u")return;const e=this.getQueryParams(this.global.location.search);Ph in e&&e[Ph].split(",").forEach(n=>{const[r,i]=n.split(":");this.urlFlags[r]=k0(r,i)})}}function v0(s){const e={};return s.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(t,...n)=>(S0(e,n[0],n[1]),n.join("="))),e}function S0(s,e,t){s[decodeURIComponent(e)]=decodeURIComponent(t||"")}function k0(s,e){const t=e.toLowerCase();return t==="true"||t==="false"?t==="true":`${+t}`===t?+t:e}function ge(){return Zd}let Zd=null;function I0(s){Zd=s}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Xl;function Qd(){if(Xl==null){let s;if(typeof window<"u")s=window;else if(typeof global<"u")s=global;else if(typeof process<"u")s=process;else if(typeof self<"u")s=self;else throw new Error("Could not find a global object");Xl=s}return Xl}function E0(){const s=Qd();return s._tfGlobals==null&&(s._tfGlobals=new Map),s._tfGlobals}function bc(s,e){const t=E0();if(t.has(s))return t.get(s);{const n=e();return t.set(s,n),t.get(s)}}const T0="Abs",Jd="Add",A0="All",C0="ArgMax",N0="AvgPool",$0="AvgPool3D",D0="BatchMatMul",O0="Bincount",ep="Cast",M0="ClipByValue",P0="Complex",R0="ComplexAbs",tp="Concat",L0="Conv2D",B0="Conv2DBackpropFilter",F0="Conv2DBackpropInput",U0="Conv3D",z0="Conv3DBackpropInputV2",W0="CropAndResize",G0="DepthwiseConv2dNative",V0="RealDiv",q0="Einsum",H0="Elu",j0="Erf",K0="Equal",X0="Exp",Y0="ExpandDims",Z0="Fill",Q0="FlipLeftRight",J0="Floor",eb="FloorDiv",tb="GatherV2",nb="Greater",sb="GreaterEqual",wc="Identity",rb="Imag",ib="LeakyRelu",ob="Less",ab="LessEqual",lb="Log",ub="Log1p",cb="LogicalAnd",hb="Max",fb="Maximum",np="MaxPool",db="MaxPool3D",pb="Mean",mb="Min",gb="Minimum",yb="MirrorPad",bb="Multiply",wb="Neg",xb="NonMaxSuppressionV3",_b="NonMaxSuppressionV4",vb="NonMaxSuppressionV5",Sb="OnesLike",kb="OneHot",Ib="Pack",sp="PadV2",Eb="Pow",Tb="Prelu",Ab="Range",Cb="Real",Nb="Relu",$b="Reshape",rp="ResizeNearestNeighbor",Db="ResizeBilinear",Ob="Relu6",Mb="Round",Pb="Select",Rb="Selu",ip="Slice",Lb="Sigmoid",Bb="Softplus",Fb="Sqrt",Ub="Sum",zb="SplitV",Wb="Softmax",Gb="Sub",Vb="Tanh",op="Tile",qb="Transform",Yl="Transpose",Hb="Unpack",jb="ZerosLike",Kb="Step",Xb="RotateWithOffset",Au="FusedConv2D";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rr(...s){ge().getBool("IS_TEST")||ge().getBool("PROD")||console.warn(...s)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ba=bc("kernelRegistry",()=>new Map),Yb=bc("gradRegistry",()=>new Map);function Rh(s,e){const t=ap(s,e);return Ba.get(t)}function Lh(s){return Yb.get(s)}function Bh(s){const e=Ba.entries(),t=[];for(;;){const{done:n,value:r}=e.next();if(n)break;const[i,o]=r,[a]=i.split("_");a===s&&t.push(o)}return t}function Zb(s){const{kernelName:e,backendName:t}=s,n=ap(e,t);Ba.has(n)&&Rr(`The kernel '${e}' for backend '${t}' is already registered`),Ba.set(n,s)}function ap(s,e){return`${e}_${s}`}/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lp(s){return s instanceof Float32Array||s instanceof Int32Array||s instanceof Uint8Array||s instanceof Uint8ClampedArray}function Qb(s){return s&&s.__esModule&&Object.prototype.hasOwnProperty.call(s,"default")?s.default:s}function Jb(s){if(Object.prototype.hasOwnProperty.call(s,"__esModule"))return s;var e=s.default;if(typeof e=="function"){var t=function n(){var r=!1;try{r=this instanceof n}catch{}return r?Reflect.construct(e,arguments,this.constructor):e.apply(this,arguments)};t.prototype=e.prototype}else t={};return Object.defineProperty(t,"__esModule",{value:!0}),Object.keys(s).forEach(function(n){var r=Object.getOwnPropertyDescriptor(s,n);Object.defineProperty(t,n,r.get?r:{enumerable:!0,get:function(){return s[n]}})}),t}var Zl,Fh;function e1(){if(Fh)return Zl;Fh=1,Zl=e;var s=null;try{s=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch{}function e(C,g,p){this.low=C|0,this.high=g|0,this.unsigned=!!p}e.prototype.__isLong__,Object.defineProperty(e.prototype,"__isLong__",{value:!0});function t(C){return(C&&C.__isLong__)===!0}e.isLong=t;var n={},r={};function i(C,g){var p,y,x;return g?(C>>>=0,(x=0<=C&&C<256)&&(y=r[C],y)?y:(p=a(C,(C|0)<0?-1:0,!0),x&&(r[C]=p),p)):(C|=0,(x=-128<=C&&C<128)&&(y=n[C],y)?y:(p=a(C,C<0?-1:0,!1),x&&(n[C]=p),p))}e.fromInt=i;function o(C,g){if(isNaN(C))return g?b:S;if(g){if(C<0)return b;if(C>=k)return D}else{if(C<=-A)return M;if(C+1>=A)return E}return C<0?o(-C,g).neg():a(C%w|0,C/w|0,g)}e.fromNumber=o;function a(C,g,p){return new e(C,g,p)}e.fromBits=a;var l=Math.pow;function u(C,g,p){if(C.length===0)throw Error("empty string");if(C==="NaN"||C==="Infinity"||C==="+Infinity"||C==="-Infinity")return S;if(typeof g=="number"?(p=g,g=!1):g=!!g,p=p||10,p<2||36<p)throw RangeError("radix");var y;if((y=C.indexOf("-"))>0)throw Error("interior hyphen");if(y===0)return u(C.substring(1),g,p).neg();for(var x=o(l(p,8)),I=S,N=0;N<C.length;N+=8){var L=Math.min(8,C.length-N),W=parseInt(C.substring(N,N+L),p);if(L<8){var X=o(l(p,L));I=I.mul(X).add(o(W))}else I=I.mul(x),I=I.add(o(W))}return I.unsigned=g,I}e.fromString=u;function c(C,g){return typeof C=="number"?o(C,g):typeof C=="string"?u(C,g):a(C.low,C.high,typeof g=="boolean"?g:C.unsigned)}e.fromValue=c;var h=65536,d=1<<24,w=h*h,k=w*w,A=k/2,m=i(d),S=i(0);e.ZERO=S;var b=i(0,!0);e.UZERO=b;var f=i(1);e.ONE=f;var v=i(1,!0);e.UONE=v;var _=i(-1);e.NEG_ONE=_;var E=a(-1,2147483647,!1);e.MAX_VALUE=E;var D=a(-1,-1,!0);e.MAX_UNSIGNED_VALUE=D;var M=a(0,-2147483648,!1);e.MIN_VALUE=M;var $=e.prototype;return $.toInt=function(){return this.unsigned?this.low>>>0:this.low},$.toNumber=function(){return this.unsigned?(this.high>>>0)*w+(this.low>>>0):this.high*w+(this.low>>>0)},$.toString=function(g){if(g=g||10,g<2||36<g)throw RangeError("radix");if(this.isZero())return"0";if(this.isNegative())if(this.eq(M)){var p=o(g),y=this.div(p),x=y.mul(p).sub(this);return y.toString(g)+x.toInt().toString(g)}else return"-"+this.neg().toString(g);for(var I=o(l(g,6),this.unsigned),N=this,L="";;){var W=N.div(I),X=N.sub(W.mul(I)).toInt()>>>0,V=X.toString(g);if(N=W,N.isZero())return V+L;for(;V.length<6;)V="0"+V;L=""+V+L}},$.getHighBits=function(){return this.high},$.getHighBitsUnsigned=function(){return this.high>>>0},$.getLowBits=function(){return this.low},$.getLowBitsUnsigned=function(){return this.low>>>0},$.getNumBitsAbs=function(){if(this.isNegative())return this.eq(M)?64:this.neg().getNumBitsAbs();for(var g=this.high!=0?this.high:this.low,p=31;p>0&&(g&1<<p)==0;p--);return this.high!=0?p+33:p+1},$.isZero=function(){return this.high===0&&this.low===0},$.eqz=$.isZero,$.isNegative=function(){return!this.unsigned&&this.high<0},$.isPositive=function(){return this.unsigned||this.high>=0},$.isOdd=function(){return(this.low&1)===1},$.isEven=function(){return(this.low&1)===0},$.equals=function(g){return t(g)||(g=c(g)),this.unsigned!==g.unsigned&&this.high>>>31===1&&g.high>>>31===1?!1:this.high===g.high&&this.low===g.low},$.eq=$.equals,$.notEquals=function(g){return!this.eq(g)},$.neq=$.notEquals,$.ne=$.notEquals,$.lessThan=function(g){return this.comp(g)<0},$.lt=$.lessThan,$.lessThanOrEqual=function(g){return this.comp(g)<=0},$.lte=$.lessThanOrEqual,$.le=$.lessThanOrEqual,$.greaterThan=function(g){return this.comp(g)>0},$.gt=$.greaterThan,$.greaterThanOrEqual=function(g){return this.comp(g)>=0},$.gte=$.greaterThanOrEqual,$.ge=$.greaterThanOrEqual,$.compare=function(g){if(t(g)||(g=c(g)),this.eq(g))return 0;var p=this.isNegative(),y=g.isNegative();return p&&!y?-1:!p&&y?1:this.unsigned?g.high>>>0>this.high>>>0||g.high===this.high&&g.low>>>0>this.low>>>0?-1:1:this.sub(g).isNegative()?-1:1},$.comp=$.compare,$.negate=function(){return!this.unsigned&&this.eq(M)?M:this.not().add(f)},$.neg=$.negate,$.add=function(g){t(g)||(g=c(g));var p=this.high>>>16,y=this.high&65535,x=this.low>>>16,I=this.low&65535,N=g.high>>>16,L=g.high&65535,W=g.low>>>16,X=g.low&65535,V=0,Z=0,te=0,oe=0;return oe+=I+X,te+=oe>>>16,oe&=65535,te+=x+W,Z+=te>>>16,te&=65535,Z+=y+L,V+=Z>>>16,Z&=65535,V+=p+N,V&=65535,a(te<<16|oe,V<<16|Z,this.unsigned)},$.subtract=function(g){return t(g)||(g=c(g)),this.add(g.neg())},$.sub=$.subtract,$.multiply=function(g){if(this.isZero())return S;if(t(g)||(g=c(g)),s){var p=s.mul(this.low,this.high,g.low,g.high);return a(p,s.get_high(),this.unsigned)}if(g.isZero())return S;if(this.eq(M))return g.isOdd()?M:S;if(g.eq(M))return this.isOdd()?M:S;if(this.isNegative())return g.isNegative()?this.neg().mul(g.neg()):this.neg().mul(g).neg();if(g.isNegative())return this.mul(g.neg()).neg();if(this.lt(m)&&g.lt(m))return o(this.toNumber()*g.toNumber(),this.unsigned);var y=this.high>>>16,x=this.high&65535,I=this.low>>>16,N=this.low&65535,L=g.high>>>16,W=g.high&65535,X=g.low>>>16,V=g.low&65535,Z=0,te=0,oe=0,ce=0;return ce+=N*V,oe+=ce>>>16,ce&=65535,oe+=I*V,te+=oe>>>16,oe&=65535,oe+=N*X,te+=oe>>>16,oe&=65535,te+=x*V,Z+=te>>>16,te&=65535,te+=I*X,Z+=te>>>16,te&=65535,te+=N*W,Z+=te>>>16,te&=65535,Z+=y*V+x*X+I*W+N*L,Z&=65535,a(oe<<16|ce,Z<<16|te,this.unsigned)},$.mul=$.multiply,$.divide=function(g){if(t(g)||(g=c(g)),g.isZero())throw Error("division by zero");if(s){if(!this.unsigned&&this.high===-2147483648&&g.low===-1&&g.high===-1)return this;var p=(this.unsigned?s.div_u:s.div_s)(this.low,this.high,g.low,g.high);return a(p,s.get_high(),this.unsigned)}if(this.isZero())return this.unsigned?b:S;var y,x,I;if(this.unsigned){if(g.unsigned||(g=g.toUnsigned()),g.gt(this))return b;if(g.gt(this.shru(1)))return v;I=b}else{if(this.eq(M)){if(g.eq(f)||g.eq(_))return M;if(g.eq(M))return f;var N=this.shr(1);return y=N.div(g).shl(1),y.eq(S)?g.isNegative()?f:_:(x=this.sub(g.mul(y)),I=y.add(x.div(g)),I)}else if(g.eq(M))return this.unsigned?b:S;if(this.isNegative())return g.isNegative()?this.neg().div(g.neg()):this.neg().div(g).neg();if(g.isNegative())return this.div(g.neg()).neg();I=S}for(x=this;x.gte(g);){y=Math.max(1,Math.floor(x.toNumber()/g.toNumber()));for(var L=Math.ceil(Math.log(y)/Math.LN2),W=L<=48?1:l(2,L-48),X=o(y),V=X.mul(g);V.isNegative()||V.gt(x);)y-=W,X=o(y,this.unsigned),V=X.mul(g);X.isZero()&&(X=f),I=I.add(X),x=x.sub(V)}return I},$.div=$.divide,$.modulo=function(g){if(t(g)||(g=c(g)),s){var p=(this.unsigned?s.rem_u:s.rem_s)(this.low,this.high,g.low,g.high);return a(p,s.get_high(),this.unsigned)}return this.sub(this.div(g).mul(g))},$.mod=$.modulo,$.rem=$.modulo,$.not=function(){return a(~this.low,~this.high,this.unsigned)},$.and=function(g){return t(g)||(g=c(g)),a(this.low&g.low,this.high&g.high,this.unsigned)},$.or=function(g){return t(g)||(g=c(g)),a(this.low|g.low,this.high|g.high,this.unsigned)},$.xor=function(g){return t(g)||(g=c(g)),a(this.low^g.low,this.high^g.high,this.unsigned)},$.shiftLeft=function(g){return t(g)&&(g=g.toInt()),(g&=63)===0?this:g<32?a(this.low<<g,this.high<<g|this.low>>>32-g,this.unsigned):a(0,this.low<<g-32,this.unsigned)},$.shl=$.shiftLeft,$.shiftRight=function(g){return t(g)&&(g=g.toInt()),(g&=63)===0?this:g<32?a(this.low>>>g|this.high<<32-g,this.high>>g,this.unsigned):a(this.high>>g-32,this.high>=0?0:-1,this.unsigned)},$.shr=$.shiftRight,$.shiftRightUnsigned=function(g){if(t(g)&&(g=g.toInt()),g&=63,g===0)return this;var p=this.high;if(g<32){var y=this.low;return a(y>>>g|p<<32-g,p>>>g,this.unsigned)}else return g===32?a(p,0,this.unsigned):a(p>>>g-32,0,this.unsigned)},$.shru=$.shiftRightUnsigned,$.shr_u=$.shiftRightUnsigned,$.toSigned=function(){return this.unsigned?a(this.low,this.high,!1):this},$.toUnsigned=function(){return this.unsigned?this:a(this.low,this.high,!0)},$.toBytes=function(g){return g?this.toBytesLE():this.toBytesBE()},$.toBytesLE=function(){var g=this.high,p=this.low;return[p&255,p>>>8&255,p>>>16&255,p>>>24,g&255,g>>>8&255,g>>>16&255,g>>>24]},$.toBytesBE=function(){var g=this.high,p=this.low;return[g>>>24,g>>>16&255,g>>>8&255,g&255,p>>>24,p>>>16&255,p>>>8&255,p&255]},e.fromBytes=function(g,p,y){return y?e.fromBytesLE(g,p):e.fromBytesBE(g,p)},e.fromBytesLE=function(g,p){return new e(g[0]|g[1]<<8|g[2]<<16|g[3]<<24,g[4]|g[5]<<8|g[6]<<16|g[7]<<24,p)},e.fromBytesBE=function(g,p){return new e(g[4]<<24|g[5]<<16|g[6]<<8|g[7],g[0]<<24|g[1]<<16|g[2]<<8|g[3],p)},Zl}var up=e1(),cp=Qb(up),t1=r0({__proto__:null,default:cp},[up]);/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const tr=cp||t1;function Al(s){return tr.fromString(s,!0,16)}const hp=Al("c3a5c85c97cb3127"),zs=Al("b492b66fbe98f273"),St=Al("9ae16a3b2f90404f");function Cu(s){return s.xor(s.shru(47))}function fp(s,e,t){const n=s.slice(e,e+t);return tr.fromBytes(Array.from(n),!0,!0)}function $e(s,e){return fp(s,e,8)}function Uh(s,e){return fp(s,e,4)}function tt(s,e){return e===0?s:s.shru(e).or(s.shl(64-e))}function $s(s,e,t=Al("9ddfea08eb382d69")){let n=s.xor(e).mul(t);n=n.xor(n.shru(47));let r=e.xor(n).mul(t);return r=r.xor(r.shru(47)),r=r.mul(t),r}function n1(s,e,t,n,r,i){r=r.add(s),i=tt(i.add(r).add(n),21);const o=r;return r=r.add(e),r=r.add(t),i=i.add(tt(r,44)),[r.add(n),i.add(o)]}function Xo(s,e,t,n){return n1($e(s,e),$e(s,e+8),$e(s,e+16),$e(s,e+24),t,n)}function s1(s,e=s.length){if(e>=8){const t=St.add(e*2),n=$e(s,0).add(St),r=$e(s,e-8),i=tt(r,37).mul(t).add(n),o=tt(n,25).add(r).mul(t);return $s(i,o,t)}if(e>=4){const t=St.add(e*2),n=Uh(s,0);return $s(n.shl(3).add(e),Uh(s,e-4),t)}if(e>0){const t=s[0],n=s[e>>1],r=s[e-1],i=t+(n<<8),o=e+(r<<2);return Cu(St.mul(i).xor(hp.mul(o))).mul(St)}return St}function r1(s,e=s.length){const t=St.add(e*2),n=$e(s,0).mul(zs),r=$e(s,8),i=$e(s,e-8).mul(t),o=$e(s,e-16).mul(St);return $s(tt(n.add(r),43).add(tt(i,30)).add(o),n.add(tt(r.add(St),18)).add(i),t)}function i1(s,e=s.length){const t=St.add(e*2),n=$e(s,0).mul(St),r=$e(s,8),i=$e(s,e-8).mul(t),o=$e(s,e-16).mul(St),a=tt(n.add(r),43).add(tt(i,30)).add(o),l=$s(a,n.add(tt(r.add(St),18)).add(i),t),u=$e(s,16).mul(t),c=$e(s,24),h=a.add($e(s,e-32)).mul(t),d=l.add($e(s,e-24)).mul(t);return $s(tt(u.add(c),43).add(tt(h,30)).add(d),u.add(tt(c.add(n),18)).add(h),t)}function o1(s,e=s.length){const t=tr.fromNumber(81,!0);if(e<=32)return e<=16?s1(s,e):r1(s,e);if(e<=64)return i1(s,e);let n=t,r=t.mul(zs).add(113),i=Cu(r.mul(St).add(113)).mul(St),o=[tr.UZERO,tr.UZERO],a=[tr.UZERO,tr.UZERO];n=n.mul(St).add($e(s,0));let l=0;const u=(e-1>>6)*64,c=u+(e-1&63)-63;do n=tt(n.add(r).add(o[0]).add($e(s,l+8)),37).mul(zs),r=tt(r.add(o[1]).add($e(s,l+48)),42).mul(zs),n=n.xor(a[1]),r=r.add(o[0]).add($e(s,l+40)),i=tt(i.add(a[0]),33).mul(zs),o=Xo(s,l,o[1].mul(zs),n.add(a[0])),a=Xo(s,l+32,i.add(a[1]),r.add($e(s,l+16))),[i,n]=[n,i],l+=64;while(l!==u);const h=zs.add(i.and(255).shl(1));return l=c,a[0]=a[0].add(e-1&63),o[0]=o[0].add(a[0]),a[0]=a[0].add(o[0]),n=tt(n.add(r).add(o[0]).add($e(s,l+8)),37).mul(h),r=tt(r.add(o[1]).add($e(s,l+48)),42).mul(h),n=n.xor(a[1].mul(9)),r=r.add(o[0].mul(9).add($e(s,l+40))),i=tt(i.add(a[0]),33).mul(h),o=Xo(s,l,o[1].mul(h),n.add(a[0])),a=Xo(s,l+32,i.add(a[1]),r.add($e(s,l+16))),[i,n]=[n,i],$s($s(o[0],a[0],h).add(Cu(r).mul(hp)).add(i),$s(o[1],a[1],h).add(n),h)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function a1(s,e){return e==="string"?fr(s):Cl([s],e)}function l1(s,e){return s instanceof Float32Array&&e==="float32"||s instanceof Int32Array&&e==="int32"||s instanceof Uint8Array&&e==="bool"}function Cl(s,e){if(e==="string")throw new Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(s)&&(s=_r(s)),ge().getBool("DEBUG")&&y0(s,e),l1(s,e))return s;if(e==null||e==="float32"||e==="complex64")return new Float32Array(s);if(e==="int32")return new Int32Array(s);if(e==="bool"){const t=new Uint8Array(s.length);for(let n=0;n<t.length;++n)Math.round(s[n])!==0&&(t[n]=1);return t}else throw new Error(`Unknown data type ${e}`)}function ui(){return ge().platform.now()}function fr(s,e="utf-8"){return e=e||"utf-8",ge().platform.encode(s,e)}function Fa(s,e="utf-8"){return e=e||"utf-8",ge().platform.decode(s,e)}function ln(s){return ge().platform.isTypedArray!=null?ge().platform.isTypedArray(s):lp(s)}function _r(s,e=[],t=!1){if(e==null&&(e=[]),typeof s=="boolean"||typeof s=="number"||typeof s=="string"||yc(s)||s==null||ln(s)&&t)e.push(s);else if(Array.isArray(s)||ln(s))for(let n=0;n<s.length;++n)_r(s[n],e,t);else{let n=-1;for(const r of Object.keys(s))/^([1-9]+[0-9]*|0)$/.test(r)&&(n=Math.max(n,Number(r)));for(let r=0;r<=n;r++)_r(s[r],e,t)}return e}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class u1{constructor(e,t){this.backendTimer=e,this.logger=t,t==null&&(this.logger=new h1)}profileKernel(e,t,n){let r;const i=()=>{r=n()};let o;const a=ui();if(this.backendTimer.timerAvailable())o=this.backendTimer.time(i);else{i();for(const u of r)u.dataSync();o=Promise.resolve({kernelMs:ui()-a})}if(ge().getBool("CHECK_COMPUTATION_FOR_ERRORS"))for(let u=0;u<r.length;u++){const c=r[u];c.data().then(h=>{c1(h,c.dtype,e)})}return{kernelName:e,outputs:r,inputs:t,timeMs:o.then(u=>u.kernelMs),extraInfo:o.then(u=>u.getExtraProfileInfo!=null?u.getExtraProfileInfo():"")}}logKernelProfile(e){const{kernelName:t,outputs:n,timeMs:r,inputs:i,extraInfo:o}=e;n.forEach(a=>{Promise.all([a.data(),r,o]).then(l=>{this.logger.logKernelProfile(t,a,l[0],l[1],i,l[2])})})}}function c1(s,e,t){if(e!=="float32")return!1;for(let n=0;n<s.length;n++){const r=s[n];if(isNaN(r)||!isFinite(r))return console.warn(`Found ${r} in the result of '${t}'`),!0}return!1}class h1{logKernelProfile(e,t,n,r,i,o){const a=typeof r=="number"?da(`${r}ms`,9):r.error,l=da(e,25),u=t.rank,c=t.size,h=da(t.shape.toString(),14);let d="";for(const w in i){const k=i[w];if(k!=null){const A=k.shape||t.shape,m=A.length;d+=`${w}: ${m}D ${m>0?A:""} `}}console.log(`%c${l}	%c${a}	%c${u}D ${h}	%c${c}	%c${d}	%c${o}`,"font-weight:bold","color:red","color:blue","color: orange","color: green","color: steelblue")}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function f1(s,e,t){const n={},r={};for(let l=0;l<e.length;l++)n[e[l].id]=!0;for(let l=0;l<s.length;l++){const u=s[l],c=u.inputs;for(const h in c){const d=c[h];let w=!1;for(let k=0;k<e.length;k++)if(n[d.id]){u.outputs.forEach(A=>n[A.id]=!0),w=!0,r[u.id]=!0;break}if(w)break}}const i={};i[t.id]=!0;const o={};for(let l=s.length-1;l>=0;l--){const u=s[l],c=u.inputs;for(let h=0;h<u.outputs.length;h++)if(i[u.outputs[h].id]){for(const d in c)i[c[d].id]=!0,o[u.id]=!0;break}}const a=[];for(let l=0;l<s.length;l++){const u=s[l];if(r[u.id]&&o[u.id]){const c={};for(const d in u.inputs){const w=u.inputs[d];n[w.id]&&(c[d]=w)}const h=Object.assign({},u);h.inputs=c,h.outputs=u.outputs,a.push(h)}}return a}function d1(s,e,t,n){for(let r=e.length-1;r>=0;r--){const i=e[r],o=[];if(i.outputs.forEach(l=>{const u=s[l.id];u!=null?o.push(u):o.push(null)}),i.gradient==null)throw new Error(`Cannot compute gradient: gradient function not found for ${i.kernelName}.`);const a=i.gradient(o);for(const l in i.inputs){if(!(l in a))throw new Error(`Cannot backprop through input ${l}. Available gradients found: ${Object.keys(a)}.`);const u=t(()=>a[l]());if(u.dtype!=="float32")throw new Error(`Error in gradient for op ${i.kernelName}. The gradient of input ${l} must have 'float32' dtype, but has '${u.dtype}'`);const c=i.inputs[l];if(!cn(u.shape,c.shape))throw new Error(`Error in gradient for op ${i.kernelName}. The gradient of input '${l}' has shape '${u.shape}', which does not match the shape of the input '${c.shape}'`);if(s[c.id]==null)s[c.id]=u;else{const h=s[c.id];s[c.id]=n(h,u),h.dispose()}}}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zh=20,Ni=3,Ql=7;function p1(s,e,t,n){const r=Zt(e),i=m1(s,e,t,r),o=e.length,a=pa(s,e,t,r,i),l=["Tensor"];return n&&(l.push(`  dtype: ${t}`),l.push(`  rank: ${o}`),l.push(`  shape: [${e}]`),l.push("  values:")),l.push(a.map(u=>"    "+u).join(`
`)),l.join(`
`)}function m1(s,e,t,n){const r=me(e),i=n[n.length-1],o=new Array(i).fill(0),a=e.length,l=t==="complex64"?Fi(s):s;if(a>1)for(let u=0;u<r/i;u++){const c=u*i;for(let h=0;h<i;h++)o[h]=Math.max(o[h],Bi(l[c+h],0,t).length)}return o}function Bi(s,e,t){let n;return Array.isArray(s)?n=`${parseFloat(s[0].toFixed(Ql))} + ${parseFloat(s[1].toFixed(Ql))}j`:Tl(s)?n=`'${s}'`:t==="bool"?n=dp(s):n=parseFloat(s.toFixed(Ql)).toString(),da(n,e)}function dp(s){return s===0?"false":"true"}function pa(s,e,t,n,r,i=!0){const o=t==="complex64"?2:1,a=e[0],l=e.length;if(l===0){if(t==="complex64"){const A=Fi(s);return[Bi(A[0],0,t)]}return t==="bool"?[dp(s[0])]:[s[0].toString()]}if(l===1){if(a>zh){const m=Ni*o;let S=Array.from(s.slice(0,m)),b=Array.from(s.slice((a-Ni)*o,a*o));return t==="complex64"&&(S=Fi(S),b=Fi(b)),["["+S.map((f,v)=>Bi(f,r[v],t)).join(", ")+", ..., "+b.map((f,v)=>Bi(f,r[a-Ni+v],t)).join(", ")+"]"]}return["["+(t==="complex64"?Fi(s):Array.from(s)).map((m,S)=>Bi(m,r[S],t)).join(", ")+"]"]}const u=e.slice(1),c=n.slice(1),h=n[0]*o,d=[];if(a>zh){for(let A=0;A<Ni;A++){const m=A*h,S=m+h;d.push(...pa(s.slice(m,S),u,t,c,r,!1))}d.push("...");for(let A=a-Ni;A<a;A++){const m=A*h,S=m+h;d.push(...pa(s.slice(m,S),u,t,c,r,A===a-1))}}else for(let A=0;A<a;A++){const m=A*h,S=m+h;d.push(...pa(s.slice(m,S),u,t,c,r,A===a-1))}const w=l===2?",":"";d[0]="["+(a>0?d[0]+w:"");for(let A=1;A<d.length-1;A++)d[A]=" "+d[A]+w;let k=`,
`;for(let A=2;A<l;A++)k+=`
`;return d[d.length-1]=" "+d[d.length-1]+"]"+(i?"":k),d}function Fi(s){const e=[];for(let t=0;t<s.length;t+=2)e.push([s[t],s[t+1]]);return e}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ua{constructor(e,t,n){if(this.dtype=t,this.shape=e.slice(),this.size=me(e),n!=null){const r=n.length;R(r===this.size,()=>`Length of values '${r}' does not match the size inferred by the shape '${this.size}'.`)}if(t==="complex64")throw new Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=n||nt(t,this.size),this.strides=Zt(e)}set(e,...t){t.length===0&&(t=[0]),R(t.length===this.rank,()=>`The number of provided coordinates (${t.length}) must match the rank (${this.rank})`);const n=this.locToIndex(t);this.values[n]=e}get(...e){e.length===0&&(e=[0]);let t=0;for(const r of e){if(r<0||r>=this.shape[t]){const i=`Requested out of range element at ${e}.   Buffer shape=${this.shape}`;throw new Error(i)}t++}let n=e[e.length-1];for(let r=0;r<e.length-1;++r)n+=this.strides[r]*e[r];return this.values[n]}locToIndex(e){if(this.rank===0)return 0;if(this.rank===1)return e[0];let t=e[e.length-1];for(let n=0;n<e.length-1;++n)t+=this.strides[n]*e[n];return t}indexToLoc(e){if(this.rank===0)return[];if(this.rank===1)return[e];const t=new Array(this.shape.length);for(let n=0;n<t.length-1;++n)t[n]=Math.floor(e/this.strides[n]),e-=t[n]*this.strides[n];return t[t.length-1]=e,t}get rank(){return this.shape.length}toTensor(){return In().makeTensor(this.values,this.shape,this.dtype)}}let In=null,Lr=null;function g1(s){In=s}function y1(s){Lr=s}class pt{constructor(e,t,n,r){this.kept=!1,this.isDisposedInternal=!1,this.shape=e.slice(),this.dtype=t||"float32",this.size=me(e),this.strides=Zt(e),this.dataId=n,this.id=r,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){const e=await this.data();return Lr.buffer(this.shape,this.dtype,e)}bufferSync(){return Lr.buffer(this.shape,this.dtype,this.dataSync())}async array(){const e=await this.data();return Mh(this.shape,e,this.dtype==="complex64")}arraySync(){return Mh(this.shape,this.dataSync(),this.dtype==="complex64")}async data(){this.throwIfDisposed();const e=In().read(this.dataId);if(this.dtype==="string"){const t=await e;try{return t.map(n=>Fa(n))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return e}dataToGPU(e){return this.throwIfDisposed(),In().readToGPU(this.dataId,e)}dataSync(){this.throwIfDisposed();const e=In().readSync(this.dataId);if(this.dtype==="string")try{return e.map(t=>Fa(t))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return e}async bytes(){this.throwIfDisposed();const e=await In().read(this.dataId);return this.dtype==="string"?e:new Uint8Array(e.buffer)}dispose(){this.isDisposed||(this.kerasMask&&this.kerasMask.dispose(),In().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw new Error("Tensor is disposed.")}print(e=!1){return Lr.print(this,e)}clone(){return this.throwIfDisposed(),Lr.clone(this)}toString(e=!1){const t=this.dataSync();return p1(t,this.shape,this.dtype,e)}cast(e){return this.throwIfDisposed(),Lr.cast(this,e)}variable(e=!0,t,n){return this.throwIfDisposed(),In().makeVariable(this,e,t,n)}}Object.defineProperty(pt,Symbol.hasInstance,{value:s=>!!s&&s.data!=null&&s.dataSync!=null&&s.throwIfDisposed!=null});function pp(){return bc("Tensor",()=>pt)}pp();class za extends pt{constructor(e,t,n,r){super(e.shape,e.dtype,e.dataId,r),this.trainable=t,this.name=n}assign(e){if(e.dtype!==this.dtype)throw new Error(`dtype of the new value (${e.dtype}) and previous value (${this.dtype}) must match`);if(!cn(e.shape,this.shape))throw new Error(`shape of the new value (${e.shape}) and previous value (${this.shape}) must match`);In().disposeTensor(this),this.dataId=e.dataId,In().incRef(this,null)}dispose(){In().disposeVariable(this),this.isDisposedInternal=!0}}Object.defineProperty(za,Symbol.hasInstance,{value:s=>s instanceof pt&&s.assign!=null&&s.assign instanceof Function});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Wh;(function(s){s.R0="R0",s.R1="R1",s.R2="R2",s.R3="R3",s.R4="R4",s.R5="R5",s.R6="R6"})(Wh||(Wh={}));var Nu;(function(s){s.float32="float32",s.int32="int32",s.bool="int32",s.complex64="complex64"})(Nu||(Nu={}));var $u;(function(s){s.float32="float32",s.int32="int32",s.bool="bool",s.complex64="complex64"})($u||($u={}));var Du;(function(s){s.float32="float32",s.int32="float32",s.bool="float32",s.complex64="complex64"})(Du||(Du={}));var Ou;(function(s){s.float32="complex64",s.int32="complex64",s.bool="complex64",s.complex64="complex64"})(Ou||(Ou={}));const b1={float32:Du,int32:Nu,bool:$u,complex64:Ou};function xc(s,e){if(s==="string"||e==="string"){if(s==="string"&&e==="string")return"string";throw new Error(`Can not upcast ${s} with ${e}`)}return b1[s][e]}function w1(s){return xc(s,"int32")}function mp(s){return s!=null&&typeof s=="object"&&"texture"in s&&s.texture instanceof WebGLTexture}function gp(s){return typeof GPUBuffer<"u"&&s!=null&&typeof s=="object"&&"buffer"in s&&s.buffer instanceof GPUBuffer}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $t(s,e){if(s.dtype===e.dtype)return[s,e];const t=xc(s.dtype,e.dtype);return[s.cast(t),e.cast(t)]}function yp(s){const e=[];return bp(s,e,new Set),e}function bp(s,e,t){if(s==null)return;if(s instanceof pt){e.push(s);return}if(!x1(s))return;const n=s;for(const r in n){const i=n[r];t.has(i)||(t.add(i),bp(i,e,t))}}function x1(s){return Array.isArray(s)||typeof s=="object"}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jl(s){return s.kernelName!=null}class Gh{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(e=>e.name)))}}}dispose(){for(const e in this.registeredVariables)this.registeredVariables[e].dispose()}}class ci{constructor(e){this.ENV=e,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new Gh}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;const e=this.getSortedBackends();for(let t=0;t<e.length;t++){const n=e[t];if(await this.initializeBackend(n).success){await this.setBackend(n);return}}throw new Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(this.pendingBackendInit!=null)throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){const{name:e,asyncInit:t}=this.initializeBackendsAndReturnBest();if(t)throw new Error(`The highest priority backend '${e}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(e)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(e){if(!(e in this.registry))if(e in this.registryFactory){const{asyncInit:t}=this.initializeBackend(e);if(t)return null}else return null;return this.registry[e]}findBackendFactory(e){return e in this.registryFactory?this.registryFactory[e].factory:null}registerBackend(e,t,n=1){return e in this.registryFactory?(Rr(`${e} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[e]={factory:t,priority:n},!0)}async setBackend(e){if(this.registryFactory[e]==null)throw new Error(`Backend name '${e}' not found in registry`);if(this.backendName=e,this.registry[e]==null){this.backendInstance=null;const{success:t,asyncInit:n}=this.initializeBackend(e);if(!(n?await t:t))return!1}return this.backendInstance=this.registry[e],this.setupRegisteredKernels(),this.profiler=new u1(this.backendInstance),!0}setupRegisteredKernels(){Bh(this.backendName).forEach(t=>{t.setupFunc!=null&&t.setupFunc(this.backendInstance)})}disposeRegisteredKernels(e){Bh(e).forEach(n=>{n.disposeFunc!=null&&n.disposeFunc(this.registry[e])})}initializeBackend(e){const t=this.registryFactory[e];if(t==null)throw new Error(`Cannot initialize backend ${e}, no registration found.`);try{const n=t.factory();if(n&&!(n instanceof jd)&&typeof n.then=="function"){const r=++this.pendingBackendInitId,i=n.then(o=>r<this.pendingBackendInitId?!1:(this.registry[e]=o,this.pendingBackendInit=null,!0)).catch(o=>(r<this.pendingBackendInitId||(this.pendingBackendInit=null,Rr(`Initialization of backend ${e} failed`),Rr(o.stack||o.message)),!1));return this.pendingBackendInit=i,{success:i,asyncInit:!0}}else return this.registry[e]=n,{success:!0,asyncInit:!1}}catch(n){return Rr(`Initialization of backend ${e} failed`),Rr(n.stack||n.message),{success:!1,asyncInit:!1}}}removeBackend(e){if(!(e in this.registryFactory))throw new Error(`${e} backend not found in registry`);this.backendName===e&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,e in this.registry&&(this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e]),delete this.registryFactory[e],this.backendName===e&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw new Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((e,t)=>this.registryFactory[t].priority-this.registryFactory[e].priority)}initializeBackendsAndReturnBest(){const e=this.getSortedBackends();for(let t=0;t<e.length;t++){const n=e[t],{success:r,asyncInit:i}=this.initializeBackend(n);if(i||r)return{name:n,asyncInit:i}}throw new Error("Could not initialize any backends, all backend initializations failed.")}moveData(e,t){const n=this.state.tensorInfo.get(t),r=n.backend,i=this.readSync(t),o=r.refCount(t);r.disposeData(t,!0),n.backend=e,e.move(t,i,n.shape,n.dtype,o),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(e,t){let n=null;if(t==null){if(typeof e!="function")throw new Error("Please provide a function to tidy()");t=e}else{if(typeof e!="string"&&!(e instanceof String))throw new Error("When calling with two arguments, the first argument to tidy() must be a string");if(typeof t!="function")throw new Error("When calling with two arguments, the 2nd argument to tidy() must be a function");n=e}let r;return this.scopedRun(()=>this.startScope(n),()=>this.endScope(r),()=>(r=t(),r instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),r))}scopedRun(e,t,n){e();try{const r=n();return t(),r}catch(r){throw t(),r}}nextTensorId(){return ci.nextTensorId++}nextVariableId(){return ci.nextVariableId++}clone(e){const t=K.runKernel(wc,{x:e}),n={x:e},r=o=>({x:()=>{const a="float32",l={x:o},u={dtype:a};return K.runKernel(ep,l,u)}}),i=[];return this.addTapeNode(this.state.activeScope.name,n,[t],r,i,{}),t}runKernel(e,t,n){if(this.backendName==null&&this.backend,!(Rh(e,this.backendName)!=null))throw new Error(`Kernel '${e}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:e,inputs:t,attrs:n})}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(e,t,n){const r=this.backend.numDataIds();let i=0;n.forEach(l=>{i+=l.dtype==="complex64"?3:1});const o=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],a=r-t-i-o;if(a>0)throw new Error(`Backend '${this.backendName}' has an internal memory leak (${a} data ids) after running '${e}'`)}runKernelFunc(e){let t,n=[];const r=this.isTapeOn(),i=this.state.numBytes,o=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let a;this.backendName==null&&this.backend;let l;const u=Jl(e)?e.kernelName:this.state.activeScope!=null?this.state.activeScope.name:"";if(Jl(e)){const{kernelName:k,inputs:A,attrs:m}=e;this.backendName==null&&this.backend;const S=Rh(k,this.backendName);R(S!=null,()=>`Cannot find registered kernel '${k}' for backend '${this.backendName}'`),a=()=>{const b=this.backend.numDataIds();l=S.kernelFunc({inputs:A,attrs:m,backend:this.backend});const f=Array.isArray(l)?l:[l];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(k,b,f);const v=f.map(_=>_.rank!=null?_:this.makeTensorFromTensorInfo(_));if(r){const _=this.getTensorsForGradient(k,A,v);n=this.saveTensorsForBackwardMode(_)}return v}}else{const{forwardFunc:k}=e,A=m=>{r&&(n=m.map(S=>this.keep(this.clone(S))))};a=()=>{const m=this.backend.numDataIds();l=this.tidy(()=>k(this.backend,A));const S=Array.isArray(l)?l:[l];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(u,m,S),S}}const{inputs:c,attrs:h}=e,d=Jl(e)?null:e.backwardsFunc;let w;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool("DEBUG")&&!this.state.profiling?t=a():(w=this.profiler.profileKernel(u,c,()=>a()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(w),t=w.outputs)}),r&&this.addTapeNode(u,c,t,d,n,h),this.state.profiling&&this.state.activeProfile.kernels.push({name:u,bytesAdded:this.state.numBytes-i,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-o,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(c).map(k=>c[k]!=null?c[k].shape:null),outputShapes:t.map(k=>k.shape),kernelTimeMs:w.timeMs,extraInfo:w.extraInfo}),Array.isArray(l)?t:t[0]}saveTensorsForBackwardMode(e){return e.map(n=>this.keep(this.clone(n)))}getTensorsForGradient(e,t,n){const r=Lh(e);if(r!=null){const i=r.inputsToSave||[],o=r.outputsToSave||[];let a;r.saveAllInputs?(R(Array.isArray(t),()=>"saveAllInputs is true, expected inputs to be an array."),a=Object.keys(t).map(u=>t[u])):a=i.map(u=>t[u]);const l=n.filter((u,c)=>o[c]);return a.concat(l)}return[]}makeTensor(e,t,n,r){if(e==null)throw new Error("Values passed to engine.makeTensor() are null");n=n||"float32",r=r||this.backend;let i=e;n==="string"&&Tl(e[0])&&(i=e.map(l=>fr(l)));const o=r.write(i,t,n),a=new pt(t,n,o,this.nextTensorId());if(this.trackTensor(a,r),n==="string"){const l=this.state.tensorInfo.get(o),u=w0(i);this.state.numBytes+=u-l.bytes,l.bytes=u}return a}makeTensorFromDataId(e,t,n,r){n=n||"float32";const i={dataId:e,shape:t,dtype:n};return this.makeTensorFromTensorInfo(i,r)}makeTensorFromTensorInfo(e,t){const{dataId:n,shape:r,dtype:i}=e,o=new pt(r,i,n,this.nextTensorId());return this.trackTensor(o,t),o}makeVariable(e,t=!0,n,r){n=n||this.nextVariableId().toString(),r!=null&&r!==e.dtype&&(e=e.cast(r));const i=new za(e,t,n,this.nextTensorId());if(this.state.registeredVariables[i.name]!=null)throw new Error(`Variable with name ${i.name} was already registered`);return this.state.registeredVariables[i.name]=i,this.incRef(i,this.backend),i}trackTensor(e,t){this.state.numTensors++,e.dtype==="string"&&this.state.numStringTensors++;let n=0;e.dtype!=="complex64"&&e.dtype!=="string"&&(n=e.size*ku(e.dtype)),this.state.numBytes+=n,this.state.tensorInfo.has(e.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(e.dataId,{backend:t||this.backend,dtype:e.dtype,shape:e.shape,bytes:n})),e instanceof za||this.track(e)}incRef(e,t){this.trackTensor(e,t),this.backend.incRef(e.dataId)}removeDataId(e,t){this.state.tensorInfo.has(e)&&this.state.tensorInfo.get(e).backend===t&&(this.state.tensorInfo.delete(e),this.state.numDataBuffers--)}disposeTensor(e){if(!this.state.tensorInfo.has(e.dataId))return;const t=this.state.tensorInfo.get(e.dataId);if(this.state.numTensors--,e.dtype==="string"&&(this.state.numStringTensors--,this.state.numBytes-=t.bytes),e.dtype!=="complex64"&&e.dtype!=="string"){const n=e.size*ku(e.dtype);this.state.numBytes-=n}t.backend.disposeData(e.dataId)&&this.removeDataId(e.dataId,t.backend)}disposeVariables(){for(const e in this.state.registeredVariables){const t=this.state.registeredVariables[e];this.disposeVariable(t)}}disposeVariable(e){this.disposeTensor(e),this.state.registeredVariables[e.name]!=null&&delete this.state.registeredVariables[e.name]}memory(){const e=this.backend.memory();return e.numTensors=this.state.numTensors,e.numDataBuffers=this.state.numDataBuffers,e.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(e.unreliable=!0,e.reasons==null&&(e.reasons=[]),e.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),e}async profile(e){this.state.profiling=!0;const t=this.state.numBytes,n=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await e(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(r=>r.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-t,this.state.activeProfile.newTensors=this.state.numTensors-n;for(const r of this.state.activeProfile.kernels)r.kernelTimeMs=await r.kernelTimeMs,r.extraInfo=await r.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(e,t,n,r,i,o){const a={id:this.state.nextTapeNodeId++,kernelName:e,inputs:t,outputs:n,saved:i},l=Lh(e);l!=null&&(r=l.gradFunc),r!=null&&(a.gradient=u=>(u=u.map((c,h)=>{if(c==null){const d=n[h],w=Rs(d.size,d.dtype);return this.makeTensor(w,d.shape,d.dtype)}return c}),r(u.length>1?u:u[0],i,o))),this.state.activeTape.push(a)}keep(e){return e.kept=!0,e}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(e){const t={track:[],name:"unnamed scope",id:this.state.nextScopeId++};e&&(t.name=e),this.state.scopeStack.push(t),this.state.activeScope=t}endScope(e){const t=yp(e),n=new Set(t.map(i=>i.id));for(let i=0;i<this.state.activeScope.track.length;i++){const o=this.state.activeScope.track[i];!o.kept&&!n.has(o.id)&&o.dispose()}const r=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],t.forEach(i=>{!i.kept&&i.scopeId===r.id&&this.track(i)})}gradients(e,t,n,r=!1){if(R(t.length>0,()=>"gradients() received an empty list of xs."),n!=null&&n.dtype!=="float32")throw new Error(`dy must have 'float32' dtype, but has '${n.dtype}'`);const i=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",e));R(i instanceof pt,()=>"The result y returned by f() must be a tensor.");const o=f1(this.state.activeTape,t,i);if(!r&&o.length===0&&t.length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{const a={};a[i.id]=n??_1(i.shape),d1(a,o,u=>this.tidy(u),v1);const l=t.map(u=>a[u.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(u=>{for(const c of u.saved)c.dispose()}),this.state.activeTape=null),{value:i,grads:l}})}customGrad(e){return R(Eu(e),()=>"The f passed in customGrad(f) must be a function."),(...t)=>{R(t.every(a=>a instanceof pt),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");let n;const r={};t.forEach((a,l)=>{r[l]=a});const i=(a,l)=>(n=e(...t,l),R(n.value instanceof pt,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),R(Eu(n.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),n.value),o=(a,l)=>{const u=n.gradFunc(a,l),c=Array.isArray(u)?u:[u];R(c.length===t.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),R(c.every(d=>d instanceof pt),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");const h={};return c.forEach((d,w)=>{h[w]=()=>d}),h};return this.runKernelFunc({forwardFunc:i,backwardsFunc:o,inputs:r})}}readSync(e){return this.state.tensorInfo.get(e).backend.readSync(e)}read(e){return this.state.tensorInfo.get(e).backend.read(e)}readToGPU(e,t){return this.state.tensorInfo.get(e).backend.readToGPU(e,t)}async time(e){const t=ui(),n=await this.backend.time(e);return n.wallMs=ui()-t,n}track(e){return this.state.activeScope!=null&&(e.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(e)),e}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new Gh;for(const e in this.registry)this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}ci.nextTensorId=0;ci.nextVariableId=0;function _1(s){const e=Yd(me(s),"float32");return K.makeTensor(e,s,"float32")}function wp(){const s=Qd();if(s._tfengine==null){const e=new _0(s);s._tfengine=new ci(e)}return I0(s._tfengine.ENV),g1(()=>s._tfengine),s._tfengine}const K=wp();function v1(s,e){const t={a:s,b:e};return K.runKernel(Jd,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nl(s,e){let t=s;if(ln(s))return e==="string"?[]:[s.length];if(mp(s)){const r=s.channels||"RGBA";return[s.height,s.width*r.length]}else if(gp(s))return[s.buffer.size/(e==null?4:ku(e))];if(!Array.isArray(s))return[];const n=[];for(;Array.isArray(t)||ln(t)&&e!=="string";)n.push(t.length),t=t[0];return Array.isArray(s)&&ge().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&xp(s,n,[]),n}function xp(s,e,t){if(t=t||[],!Array.isArray(s)&&!ln(s)){R(e.length===0,()=>`Element arr[${t.join("][")}] is a primitive, but should be an array/TypedArray of ${e[0]} elements`);return}R(e.length>0,()=>`Element arr[${t.join("][")}] should be a primitive, but is an array of ${s.length} elements`),R(s.length===e[0],()=>`Element arr[${t.join("][")}] should have ${e[0]} elements, but has ${s.length} elements`);const n=e.slice(1);for(let r=0;r<s.length;++r)xp(s[r],n,t.concat(r))}function Vh(s,e,t,n){if(s!=="string_or_numeric"){if(s==null)throw new Error("Expected dtype cannot be null.");if(s!=="numeric"&&s!==e||s==="numeric"&&e==="string")throw new Error(`Argument '${t}' passed to '${n}' must be ${s} tensor, but got ${e} tensor`)}}function H(s,e,t,n="numeric"){if(s instanceof pp())return Vh(n,s.dtype,e,t),s;let r=No(s);if(r!=="string"&&["bool","int32","float32"].indexOf(n)>=0&&(r=n),Vh(n,r,e,t),s==null||!ln(s)&&!Array.isArray(s)&&typeof s!="number"&&typeof s!="boolean"&&typeof s!="string"){const l=s==null?"null":s.constructor.name;throw new Error(`Argument '${e}' passed to '${t}' must be a Tensor or TensorLike, but got '${l}'`)}const i=Nl(s,r);!ln(s)&&!Array.isArray(s)&&(s=[s]);const a=r!=="string"?Cl(s,r):_r(s,[],!0);return K.makeTensor(a,i,r)}function _p(s,e,t,n="numeric"){if(!Array.isArray(s))throw new Error(`Argument ${e} passed to ${t} must be a \`Tensor[]\` or \`TensorLike[]\``);return s.map((i,o)=>H(i,`${e}[${o}]`,t,n))}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $l(s,e,t,n){if(n==null)n=No(s);else if(n==="complex64")throw new Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if(gp(s)||mp(s)){if(n!=="float32"&&n!=="int32")throw new Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${n}.`);return K.backend.createTensorFromGPUData(s,e||t,n)}if(!ln(s)&&!Array.isArray(s)&&typeof s!="number"&&typeof s!="boolean"&&typeof s!="string")throw new Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(e!=null){bs(e);const r=me(e),i=me(t);R(r===i,()=>`Based on the provided shape, [${e}], the tensor should have ${r} values but has ${i}`);for(let o=0;o<t.length;++o){const a=t[o],l=o===t.length-1?a!==me(e.slice(o)):!0;R(t[o]===e[o]||!l,()=>`Error creating a new Tensor. Inferred shape (${t}) does not match the provided shape (${e}). `)}}return!ln(s)&&!Array.isArray(s)&&(s=[s]),e=e||t,s=n!=="string"?Cl(s,n):_r(s,[],!0),K.makeTensor(s,e,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ma(s,e,t){const n=Nl(s,t);return $l(s,e,n,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function At(s,e){Kd(s);const t=Nl(s,e);if(t.length!==1)throw new Error("tensor1d() requires values to be a flat/TypedArray");return $l(s,null,t,e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const S1="__op";function J(s){const e=Object.keys(s);if(e.length!==1)throw new Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${e.length} keys.`);let t=e[0];const n=s[t];t.endsWith("_")&&(t=t.substring(0,t.length-1)),t=t+S1;const r=(...i)=>{K.startScope(t);try{const o=n(...i);return yc(o)&&console.error("Cannot return a Promise inside of tidy."),K.endScope(o),o}catch(o){throw K.endScope(null),o}};return Object.defineProperty(r,"name",{value:t,configurable:!0}),r}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function k1(s,e,t=0){const n=H(s,"x","pad");if(n.rank===0)throw new Error("pad(scalar) is not defined. Pass non-scalar to pad");const r={paddings:e,constantValue:t},i={x:n};return K.runKernel(sp,i,r)}const I1=J({pad_:k1});function E1(s,e,t=0){return R(e.length===4&&e[0].length===2&&e[1].length===2&&e[2].length===2&&e[3].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),I1(s,e,t)}const T1=J({pad4d_:E1});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function A1(s,e,t){const n=H(s,"x","slice","string_or_numeric");if(n.rank===0)throw new Error("Slicing scalar is not possible");const r={x:n},i={begin:e,size:t};return K.runKernel(ip,r,i)}const dt=J({slice_:A1});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function C1(s,e,t){const n=H(s,"x","slice4d");return R(n.rank===4,()=>`slice4d expects a rank-4 tensor, but got a rank-${n.rank} tensor`),dt(n,e,t)}const bo=J({slice4d_:C1});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function N1(s){const t={x:H(s,"x","clone","string_or_numeric")};return K.runKernel(wc,t)}const dr=J({clone_:N1});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $1(s,e=0){R(s.length>=1,()=>"Pass at least one tensor to concat");const t=_p(s,"tensors","concat","string_or_numeric");if(t[0].dtype==="complex64"&&t.forEach(i=>{if(i.dtype!=="complex64")throw new Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${i.dtype}. `)}),t.length===1)return dr(t[0]);const n=t,r={axis:e};return K.runKernel(tp,n,r)}const pr=J({concat_:$1});function D1(s,e){return pr(s,e)}const O1=J({concat4d_:D1});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function M1(){return typeof window<"u"&&window.document!=null||typeof WorkerGlobalScope<"u"}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Nt=ge();Nt.registerFlag("DEBUG",()=>!1,s=>{s&&console.warn("Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.")});Nt.registerFlag("IS_BROWSER",()=>M1());Nt.registerFlag("IS_NODE",()=>typeof process<"u"&&typeof process.versions<"u"&&typeof process.versions.node<"u");Nt.registerFlag("IS_CHROME",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor));Nt.registerFlag("IS_SAFARI",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor));Nt.registerFlag("PROD",()=>!1);Nt.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>Nt.getBool("DEBUG"));Nt.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0);Nt.registerFlag("IS_TEST",()=>!1);Nt.registerFlag("CHECK_COMPUTATION_FOR_ERRORS",()=>Nt.getBool("DEBUG"));Nt.registerFlag("WRAP_TO_IMAGEBITMAP",()=>!1);Nt.registerFlag("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU",()=>!1);Nt.registerFlag("USE_SETTIMEOUTCUSTOM",()=>!1);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function P1(s,e){const t=H(s,"real","complex"),n=H(e,"imag","complex");p0(t.shape,n.shape,`real and imag shapes, ${t.shape} and ${n.shape}, must match in call to tf.complex().`);const r={real:t,imag:n};return K.runKernel(P0,r)}const _c=J({complex_:P1});class yi{static join(e){return new yi(e).slice()}constructor(e){if(this.shards=[],this.previousShardIndex=0,e==null||(e instanceof Array||(e=[e]),e=e.map(n=>ln(n)?n.buffer:n),e.length===0))return;this.bufferUniformSize=e[0].byteLength;let t=0;for(let n=0;n<e.length;n++){const r=e[n];n!==e.length-1&&r.byteLength!==this.bufferUniformSize&&(this.bufferUniformSize=void 0);const i=t+r.byteLength;this.shards.push({buffer:r,start:t,end:i}),t=i}this.shards.length===0&&(this.byteLength=0),this.byteLength=this.shards[this.shards.length-1].end}slice(e=0,t=this.byteLength){if(this.shards.length===0)return new ArrayBuffer(0);if(e=isNaN(Number(e))?0:e,t=isNaN(Number(t))?0:t,e=Math.max(0,e),t=Math.min(this.byteLength,t),t<=e)return new ArrayBuffer(0);const n=this.findShardForByte(e);if(n===-1)throw new Error(`Could not find start shard for byte ${e}`);const r=t-e,i=new ArrayBuffer(r),o=new Uint8Array(i);let a=0;for(let l=n;l<this.shards.length;l++){const u=this.shards[l],h=e+a-u.start,d=a,k=Math.min(t,u.end)-u.start,A=new Uint8Array(u.buffer,h,k-h);if(o.set(A,d),a+=A.length,t<u.end)break}return i}findShardForByte(e){if(this.shards.length===0||e<0||e>=this.byteLength)return-1;if(this.bufferUniformSize!=null)return this.previousShardIndex=Math.floor(e/this.bufferUniformSize),this.previousShardIndex;function t(r){return e<r.start?-1:e>=r.end?1:0}if(t(this.shards[this.previousShardIndex])===0)return this.previousShardIndex;const n=R1(this.shards,t);return n===-1?-1:(this.previousShardIndex=n,this.previousShardIndex)}}function R1(s,e){let t=0,n=s.length;for(;t<=n;){const r=Math.floor((n-t)/2)+t,i=e(s[r]);if(i===0)return r;i<0?n=r:t=r+1}return-1}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eu(){return K}function qh(){return K.memory()}function Q(s,e){return K.tidy(s,e)}function Pe(s){yp(s).forEach(t=>t.dispose())}function oi(s){return K.keep(s)}function L1(s,e,t=1){return K.registerBackend(s,e,t)}function B1(){return K.backend}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Hh=4;async function jh(s,e){const t=[],n=[],r=Array.isArray(s)?s.map(o=>o.name):Object.keys(s);for(let o=0;o<r.length;++o){const a=r[o],l=Array.isArray(s)?s[o].tensor:s[a];if(l.dtype!=="float32"&&l.dtype!=="int32"&&l.dtype!=="bool"&&l.dtype!=="string"&&l.dtype!=="complex64")throw new Error(`Unsupported dtype in weight '${a}': ${l.dtype}`);const u={name:a,shape:l.shape,dtype:l.dtype};if(l.dtype==="string"){const c=new Promise(async h=>{const d=await l.bytes(),w=d.reduce((m,S)=>m+S.length,0)+Hh*d.length,k=new Uint8Array(w);let A=0;for(let m=0;m<d.length;m++){const S=d[m],b=new Uint8Array(new Uint32Array([S.length]).buffer);k.set(b,A),A+=Hh,k.set(S,A),A+=S.length}h(k)});n.push(c)}else n.push(l.data());e!=null&&(u.group=e),t.push(u)}const i=await Promise.all(n);return{data:F1(i),specs:t}}function F1(s){if(s===null)throw new Error(`Invalid input value: ${JSON.stringify(s)}`);let e=0;const t=[];s.forEach(i=>{if(e+=i.byteLength,t.push(i.byteLength===i.buffer.byteLength?i:new i.constructor(i)),!(i instanceof Float32Array||i instanceof Int32Array||i instanceof Uint8Array))throw new Error(`Unsupported TypedArray subtype: ${i.constructor.name}`)});const n=new Uint8Array(e);let r=0;return t.forEach(i=>{n.set(new Uint8Array(i.buffer),r),r+=i.byteLength}),n.buffer}const vc=typeof Buffer<"u"&&(typeof Blob>"u"||typeof atob>"u"||typeof btoa>"u");function Kh(s){return vc?Buffer.byteLength(s,"utf8"):new Blob([s]).size}function U1(s){if(vc)return Buffer.from(s).toString("base64");const e=new Uint8Array(s);let t="";for(let n=0,r=e.length;n<r;n++)t+=String.fromCharCode(e[n]);return btoa(t)}function z1(s){if(vc){const n=Buffer.from(s,"base64");return n.buffer.slice(n.byteOffset,n.byteOffset+n.byteLength)}const e=atob(s),t=new Uint8Array(e.length);for(let n=0;n<e.length;++n)t.set([e.charCodeAt(n)],n);return t.buffer}function W1(s){return yi.join(s)}function vp(s){if(s.modelTopology instanceof ArrayBuffer)throw new Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:s.modelTopology==null?0:Kh(JSON.stringify(s.modelTopology)),weightSpecsBytes:s.weightSpecs==null?0:Kh(JSON.stringify(s.weightSpecs)),weightDataBytes:s.weightData==null?0:new yi(s.weightData).byteLength}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class vt{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return vt.instance==null&&(vt.instance=new vt),vt.instance}static registerSaveRouter(e){vt.getInstance().saveRouters.push(e)}static registerLoadRouter(e){vt.getInstance().loadRouters.push(e)}static getSaveHandlers(e){return vt.getHandlers(e,"save")}static getLoadHandlers(e,t){return vt.getHandlers(e,"load",t)}static getHandlers(e,t,n){const r=[];return(t==="load"?vt.getInstance().loadRouters:vt.getInstance().saveRouters).forEach(o=>{const a=o(e,n);a!==null&&r.push(a)}),r}}const G1=s=>vt.getSaveHandlers(s);/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Mu="tensorflowjs",Pu=1,ir="models_store",As="model_info_store";function Sp(){if(!ge().getBool("IS_BROWSER"))throw new Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");const s=typeof window>"u"?self:window,e=s.indexedDB||s.mozIndexedDB||s.webkitIndexedDB||s.msIndexedDB||s.shimIndexedDB;if(e==null)throw new Error("The current browser does not appear to support IndexedDB.");return e}function Ru(s){const e=s.result;e.createObjectStore(ir,{keyPath:"modelPath"}),e.createObjectStore(As,{keyPath:"modelPath"})}class vr{constructor(e){if(this.indexedDB=Sp(),e==null||!e)throw new Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=e}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,e)}async load(){return this.databaseAction(this.modelPath)}databaseAction(e,t){return new Promise((n,r)=>{const i=this.indexedDB.open(Mu,Pu);i.onupgradeneeded=()=>Ru(i),i.onsuccess=()=>{const o=i.result;if(t==null){const a=o.transaction(ir,"readonly"),u=a.objectStore(ir).get(this.modelPath);u.onsuccess=()=>{if(u.result==null)return o.close(),r(new Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));n(u.result.modelArtifacts)},u.onerror=c=>(o.close(),r(u.error)),a.oncomplete=()=>o.close()}else{t.weightData=yi.join(t.weightData);const a=vp(t),l=o.transaction(As,"readwrite");let u=l.objectStore(As),c;try{c=u.put({modelPath:this.modelPath,modelArtifactsInfo:a})}catch(d){return r(d)}let h;c.onsuccess=()=>{h=o.transaction(ir,"readwrite");const d=h.objectStore(ir);let w;try{w=d.put({modelPath:this.modelPath,modelArtifacts:t,modelArtifactsInfo:a})}catch(k){return r(k)}w.onsuccess=()=>n({modelArtifactsInfo:a}),w.onerror=k=>{u=l.objectStore(As);const A=u.delete(this.modelPath);A.onsuccess=()=>(o.close(),r(w.error)),A.onerror=m=>(o.close(),r(w.error))}},c.onerror=d=>(o.close(),r(c.error)),l.oncomplete=()=>{h==null?o.close():h.oncomplete=()=>o.close()}}},i.onerror=o=>r(i.error)})}}vr.URL_SCHEME="indexeddb://";const kp=s=>ge().getBool("IS_BROWSER")&&!Array.isArray(s)&&s.startsWith(vr.URL_SCHEME)?V1(s.slice(vr.URL_SCHEME.length)):null;vt.registerSaveRouter(kp);vt.registerLoadRouter(kp);function V1(s){return new vr(s)}function q1(s){return s.startsWith(vr.URL_SCHEME)?s.slice(vr.URL_SCHEME.length):s}class H1{constructor(){this.indexedDB=Sp()}async listModels(){return new Promise((e,t)=>{const n=this.indexedDB.open(Mu,Pu);n.onupgradeneeded=()=>Ru(n),n.onsuccess=()=>{const r=n.result,i=r.transaction(As,"readonly"),a=i.objectStore(As).getAll();a.onsuccess=()=>{const l={};for(const u of a.result)l[u.modelPath]=u.modelArtifactsInfo;e(l)},a.onerror=l=>(r.close(),t(a.error)),i.oncomplete=()=>r.close()},n.onerror=r=>t(n.error)})}async removeModel(e){return e=q1(e),new Promise((t,n)=>{const r=this.indexedDB.open(Mu,Pu);r.onupgradeneeded=()=>Ru(r),r.onsuccess=()=>{const i=r.result,o=i.transaction(As,"readwrite"),a=o.objectStore(As),l=a.get(e);let u;l.onsuccess=()=>{if(l.result==null)return i.close(),n(new Error(`Cannot find model with path '${e}' in IndexedDB.`));{const c=a.delete(e),h=()=>{u=i.transaction(ir,"readwrite");const w=u.objectStore(ir).delete(e);w.onsuccess=()=>t(l.result.modelArtifactsInfo),w.onerror=k=>n(l.error)};c.onsuccess=h,c.onerror=d=>(h(),i.close(),n(l.error))}},l.onerror=c=>(i.close(),n(l.error)),o.oncomplete=()=>{u==null?i.close():u.oncomplete=()=>i.close()}},r.onerror=i=>n(r.error)})}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ds="/",Br="tensorflowjs_models",Ip="info",j1="model_topology",K1="weight_specs",X1="weight_data",Y1="model_metadata";function Ep(s){return{info:[Br,s,Ip].join(ds),topology:[Br,s,j1].join(ds),weightSpecs:[Br,s,K1].join(ds),weightData:[Br,s,X1].join(ds),modelMetadata:[Br,s,Y1].join(ds)}}function Tp(s){for(const e of Object.values(s))window.localStorage.removeItem(e)}function Z1(s){const e=s.split(ds);if(e.length<3)throw new Error(`Invalid key format: ${s}`);return e.slice(1,e.length-1).join(ds)}function Q1(s){return s.startsWith(Sr.URL_SCHEME)?s.slice(Sr.URL_SCHEME.length):s}class Sr{constructor(e){if(!ge().getBool("IS_BROWSER")||typeof window>"u"||typeof window.localStorage>"u")throw new Error("The current environment does not support local storage.");if(this.LS=window.localStorage,e==null||!e)throw new Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=e,this.keys=Ep(this.modelPath)}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{const t=JSON.stringify(e.modelTopology),n=JSON.stringify(e.weightSpecs),r=vp(e),i=yi.join(e.weightData);try{this.LS.setItem(this.keys.info,JSON.stringify(r)),this.LS.setItem(this.keys.topology,t),this.LS.setItem(this.keys.weightSpecs,n),this.LS.setItem(this.keys.weightData,U1(i));const o={format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,signature:e.signature!=null?e.signature:void 0,userDefinedMetadata:e.userDefinedMetadata!=null?e.userDefinedMetadata:void 0,modelInitializer:e.modelInitializer!=null?e.modelInitializer:void 0,initializerSignature:e.initializerSignature!=null?e.initializerSignature:void 0,trainingConfig:e.trainingConfig!=null?e.trainingConfig:void 0};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(o)),{modelArtifactsInfo:r}}catch{throw Tp(this.keys),new Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${r.modelTopologyBytes}, weightSpecsBytes=${r.weightSpecsBytes}, weightDataBytes=${r.weightDataBytes}.`)}}}async load(){const e=JSON.parse(this.LS.getItem(this.keys.info));if(e==null)throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);if(e.modelTopologyType!=="JSON")throw new Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");const t={},n=JSON.parse(this.LS.getItem(this.keys.topology));if(n==null)throw new Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);t.modelTopology=n;const r=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(r==null)throw new Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);t.weightSpecs=r;const i=this.LS.getItem(this.keys.modelMetadata);if(i!=null){const a=JSON.parse(i);t.format=a.format,t.generatedBy=a.generatedBy,t.convertedBy=a.convertedBy,a.signature!=null&&(t.signature=a.signature),a.userDefinedMetadata!=null&&(t.userDefinedMetadata=a.userDefinedMetadata),a.modelInitializer!=null&&(t.modelInitializer=a.modelInitializer),a.initializerSignature!=null&&(t.initializerSignature=a.initializerSignature),a.trainingConfig!=null&&(t.trainingConfig=a.trainingConfig)}const o=this.LS.getItem(this.keys.weightData);if(o==null)throw new Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return t.weightData=z1(o),t}}Sr.URL_SCHEME="localstorage://";const Ap=s=>ge().getBool("IS_BROWSER")&&!Array.isArray(s)&&s.startsWith(Sr.URL_SCHEME)?J1(s.slice(Sr.URL_SCHEME.length)):null;vt.registerSaveRouter(Ap);vt.registerLoadRouter(Ap);function J1(s){return new Sr(s)}class ew{constructor(){R(ge().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),R(typeof window>"u"||typeof window.localStorage<"u",()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){const e={},t=Br+ds,n=ds+Ip;for(let r=0;r<this.LS.length;++r){const i=this.LS.key(r);if(i.startsWith(t)&&i.endsWith(n)){const o=Z1(i);e[o]=JSON.parse(this.LS.getItem(i))}}return e}async removeModel(e){e=Q1(e);const t=Ep(e);if(this.LS.getItem(t.info)==null)throw new Error(`Cannot find model at path '${e}'`);const n=JSON.parse(this.LS.getItem(t.info));return Tp(t),n}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xh="://";class qn{constructor(){this.managers={}}static getInstance(){return qn.instance==null&&(qn.instance=new qn),qn.instance}static registerManager(e,t){R(e!=null,()=>"scheme must not be undefined or null."),e.endsWith(Xh)&&(e=e.slice(0,e.indexOf(Xh))),R(e.length>0,()=>"scheme must not be an empty string.");const n=qn.getInstance();R(n.managers[e]==null,()=>`A model store manager is already registered for scheme '${e}'.`),n.managers[e]=t}static getManager(e){const t=qn.getInstance().managers[e];if(t==null)throw new Error(`Cannot find model manager for scheme '${e}'`);return t}static getSchemes(){return Object.keys(qn.getInstance().managers)}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class tw{constructor(){this.messageName="setTimeoutCustom",this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(e,t){return fetch(e,t)}now(){return performance.now()}encode(e,t){if(t!=="utf-8"&&t!=="utf8")throw new Error(`Browser's encoder only supports utf-8, but got ${t}`);return this.textEncoder==null&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(e)}decode(e,t){return new TextDecoder(t).decode(e)}setTimeoutCustom(e,t){if(typeof window>"u"||!ge().getBool("USE_SETTIMEOUTCUSTOM")){setTimeout(e,t);return}this.functionRefs.push(e),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},"*")},t),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener("message",n=>{if(n.source===window&&n.data.name===this.messageName){n.stopPropagation();const r=this.functionRefs[n.data.index];r(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}isTypedArray(e){return lp(e)}}if(ge().get("IS_BROWSER")){ge().setPlatform("browser",new tw);try{qn.registerManager(Sr.URL_SCHEME,new ew)}catch{}try{qn.registerManager(vr.URL_SCHEME,new H1)}catch{}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const nw={importFetch:()=>require("node-fetch")};let tu;class sw{constructor(){this.util=require("util"),this.textEncoder=new this.util.TextEncoder}fetch(e,t){return ge().global.fetch!=null?ge().global.fetch(e,t):(tu==null&&(tu=nw.importFetch()),tu(e,t))}now(){const e=process.hrtime();return e[0]*1e3+e[1]/1e6}encode(e,t){if(t!=="utf-8"&&t!=="utf8")throw new Error(`Node built-in encoder only supports utf-8, but got ${t}`);return this.textEncoder.encode(e)}decode(e,t){return e.length===0?"":new this.util.TextDecoder(t).decode(e)}isTypedArray(e){return this.util.types.isFloat32Array(e)||this.util.types.isInt32Array(e)||this.util.types.isUint8Array(e)||this.util.types.isUint8ClampedArray(e)}}ge().get("IS_NODE")&&!ge().get("IS_BROWSER")&&ge().setPlatform("node",new sw);/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ct(s,e="float32",t){return e=e||"float32",bs(s),new Ua(s,e,t)}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rw(s,e){const t=H(s,"x","cast");if(!b0(e))throw new Error(`Failed to cast to unknown dtype ${e}`);if(e==="string"&&t.dtype!=="string"||e!=="string"&&t.dtype==="string")throw new Error("Only strings can be casted to strings");const n={x:t},r={dtype:e};return K.runKernel(ep,n,r)}const De=J({cast_:rw});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function iw(s,e=!1){console.log(s.toString(e))}/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */wp();const ow={buffer:ct,cast:De,clone:dr,print:iw};y1(ow);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function aw(s,e){let t=H(s,"a","add"),n=H(e,"b","add");[t,n]=$t(t,n);const r={a:t,b:n};return K.runKernel(Jd,r)}const he=J({add_:aw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lw(s,e){let t=H(s,"a","floorDiv"),n=H(e,"b","floorDiv");[t,n]=$t(t,n);const r={a:t,b:n};return K.runKernel(eb,r)}const uw=J({floorDiv_:lw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cw(s,e){let t=H(s,"a","div"),n=H(e,"b","div");if([t,n]=$t(t,n),t.dtype==="int32"&&n.dtype==="int32")return uw(t,n);const r={a:t,b:n},i={};return K.runKernel(V0,r,i)}const _e=J({div_:cw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hw(s,e){let t=H(s,"a","mul"),n=H(e,"b","mul");[t,n]=$t(t,n);const r={a:t,b:n};return K.runKernel(bb,r)}const ne=J({mul_:hw});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fw(s){const e=H(s,"x","abs");if(e.dtype==="complex64"){const t={x:e};return K.runKernel(R0,t)}else{const t={x:e};return K.runKernel(T0,t)}}const Tt=J({abs_:fw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dw(s,e=null,t=!1){const r={x:H(s,"x","all","bool")},i={axis:e,keepDims:t};return K.runKernel(A0,r,i)}const pw=J({all_:dw});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mw(s,e=0){const n={x:H(s,"x","argMax")},r={axis:e};return K.runKernel(C0,n,r)}const Wa=J({argMax_:mw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gw(s,e,t,n,r,i,o="channelsLast"){const[a,l]=wo(e);let u;if(o==="channelsLast")u=[a,l,s[3],s[3]];else if(o==="channelsFirst")u=[a,l,s[1],s[1]];else throw new Error(`Unknown dataFormat ${o}`);return Sc(s,u,t,n,r,i,!1,o)}function Sc(s,e,t,n,r,i,o=!1,a="channelsLast"){let[l,u,c,h]=[-1,-1,-1,-1];if(a==="channelsLast")[l,u,c,h]=s;else if(a==="channelsFirst")[l,h,u,c]=s;else throw new Error(`Unknown dataFormat ${a}`);const[d,w,,k]=e,[A,m]=wo(t),[S,b]=wo(n),f=Lu(d,S),v=Lu(w,b),{padInfo:_,outHeight:E,outWidth:D}=ww(r,u,c,A,m,f,v,i,a),M=o?k*h:k;let $;return a==="channelsFirst"?$=[l,M,E,D]:a==="channelsLast"&&($=[l,E,D,M]),{batchSize:l,dataFormat:a,inHeight:u,inWidth:c,inChannels:h,outHeight:E,outWidth:D,outChannels:M,padInfo:_,strideHeight:A,strideWidth:m,filterHeight:d,filterWidth:w,effectiveFilterHeight:f,effectiveFilterWidth:v,dilationHeight:S,dilationWidth:b,inShape:s,outShape:$,filterShape:e}}function yw(s,e,t,n,r){n==null&&(n=bw(s,e,t));const i=s[0],o=s[1],a=Ga((i-e+2*n)/t+1,r),l=Ga((o-e+2*n)/t+1,r);return[a,l]}function bw(s,e,t,n=1){const r=Lu(e,n);return Math.floor((s[0]*(t-1)-t+r)/2)}function wo(s){return typeof s=="number"?[s,s,s]:s.length===2?[s[0],s[1],1]:s}function Lu(s,e){return e<=1?s:s+(s-1)*(e-1)}function ww(s,e,t,n,r,i,o,a,l){let u,c,h;if(typeof s=="number"){u={top:s,bottom:s,left:s,right:s,type:s===0?"VALID":"NUMBER"};const w=yw([e,t],i,n,s,a);c=w[0],h=w[1]}else if(s==="same"){c=Math.ceil(e/n),h=Math.ceil(t/r);const d=Math.max(0,(c-1)*n+i-e),w=Math.max(0,(h-1)*r+o-t),k=Math.floor(d/2),A=d-k,m=Math.floor(w/2),S=w-m;u={top:k,bottom:A,left:m,right:S,type:"SAME"}}else if(s==="valid")u={top:0,bottom:0,left:0,right:0,type:"VALID"},c=Math.ceil((e-i+1)/n),h=Math.ceil((t-o+1)/r);else if(typeof s=="object"){const d=l==="channelsLast"?s[1][0]:s[2][0],w=l==="channelsLast"?s[1][1]:s[2][1],k=l==="channelsLast"?s[2][0]:s[3][0],A=l==="channelsLast"?s[2][1]:s[3][1];u={top:d,bottom:w,left:k,right:A,type:d===0&&w===0&&k===0&&A===0?"VALID":"EXPLICIT"},c=Ga((e-i+d+w)/n+1,a),h=Ga((t-o+k+A)/r+1,a)}else throw Error(`Unknown padding parameter: ${s}`);return{padInfo:u,outHeight:c,outWidth:h}}function Ga(s,e){if(!e)return Math.trunc(s);switch(e){case"round":return Math.round(s);case"ceil":return Math.ceil(s);case"floor":return Math.floor(s);default:throw new Error(`Unknown roundingMode ${e}`)}}function Bu(s){const[e,t,n]=wo(s);return e===1&&t===1&&n===1}function bi(s,e){return Bu(s)||Bu(e)}function hi(s){return wo(s).every(e=>e>0)}function xw(s){if(s==="NHWC")return"channelsLast";if(s==="NCHW")return"channelsFirst";throw new Error(`Unknown dataFormat ${s}`)}function ts(s,e,t){if(t!=null){if(typeof e=="string")throw Error(`Error in ${s}: pad must be an integer when using dimRoundingMode ${t} but got pad ${e}.`);if(typeof e=="number")R(Su(e),()=>`Error in ${s}: pad must be an integer when using dimRoundingMode ${t} but got pad ${e}.`);else if(typeof e=="object")e.forEach(n=>{n.forEach(r=>{R(Su(r),()=>`Error in ${s}: pad must be an integer when using dimRoundingMode ${t} but got pad ${r}.`)})});else throw Error(`Error in ${s}: Unknown padding parameter: ${e}`)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _w(s,e){const n={x:H(s,"x","reshape","string_or_numeric")},r={shape:e};return K.runKernel($b,n,r)}const ae=J({reshape_:_w});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vw(s,e,t,n,r){const i=H(s,"x","avgPool","float32"),o=1;R(bi(t,o),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${t} and dilations '${o}'`);let a=i,l=!1;i.rank===3&&(l=!0,a=ae(i,[1,i.shape[0],i.shape[1],i.shape[2]])),R(a.rank===4,()=>`Error in avgPool: x must be rank 4 but got rank ${a.rank}.`),ts("avgPool",n,r);const u={x:a},c={filterSize:e,strides:t,pad:n,dimRoundingMode:r};let h=K.runKernel(N0,u,c);return h=De(h,i.dtype),l?ae(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const Sw=J({avgPool_:vw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kw(s,e,t,n,r,i="NDHWC"){const o=H(s,"x","avgPool3d","float32");let a=o,l=!1;o.rank===4&&(l=!0,a=ae(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),R(a.rank===5,()=>`Error in avgPool3d: x must be rank 5 but got rank ${a.rank}.`),R(i==="NDHWC",()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${i}`),R(typeof t=="number"&&t>0||Array.isArray(t)&&t[0]>0&&t[1]>0&&t[2]>0,()=>`Error in avgPool3d: Stride must be > 0, but got '${t}'`),ts("avgPool3d",n,r);const u={x:a},c={filterSize:e,strides:t,pad:n,dimRoundingMode:r,dataFormat:i};let h=K.runKernel($0,u,c);return h=De(h,a.dtype),l?ae(h,[h.shape[1],h.shape[2],h.shape[3],h.shape[4]]):h}const Iw=J({avgPool3d_:kw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ew(s,e,t=!1,n=!1){let r=H(s,"a","matMul"),i=H(e,"b","matMul");[r,i]=$t(r,i);const o={a:r,b:i},a={transposeA:t,transposeB:n};return K.runKernel(D0,o,a)}const Pn=J({matMul_:Ew});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tw(s){const t={x:H(s,"x","sigmoid","float32")};return K.runKernel(Lb,t)}const kc=J({sigmoid_:Tw});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Aw(s){const t={x:H(s,"x","tanh","float32")};return K.runKernel(Vb,t)}const Ic=J({tanh_:Aw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cw(s,e,t){const n=H(s,"x","bincount"),r=H(e,"weights","bincount");R(n.dtype==="int32",()=>`Error in bincount: input dtype must be int32, but got ${n.dtype}`),R(t>=0,()=>`size must be non-negative, but got ${t}.`),R(r.size===n.size||r.size===0,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${n.shape}, weights shape: ${r.shape}.`);const i={x:n,weights:r},o={size:t};return K.runKernel(O0,i,o)}const Nw=J({bincount_:Cw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $w(s,e){let t=H(s,"broadcastTo","x");const n=t.shape;if(bs(e),e.length<t.rank)throw new Error(`broadcastTo(): shape.length=${e.length} < input.rank=${t.rank}.`);if(e.length>t.rank){const u=t.shape.slice();for(;u.length<e.length;)u.unshift(1);t=ae(t,u)}const r=t.shape,i=Array.from(e);for(let u=e.length-1;u>=0;u--)if(r[u]===e[u])i[u]=1;else if(t.shape[u]!==1)throw new Error(`broadcastTo(): [${n}] cannot be broadcast to [${e}].`);if(i.map((u,c)=>u>1?c:-1).filter(u=>u>=0).length===0)return dr(t);const a={x:t},l={reps:i};return K.runKernel(op,a,l)}const ga=J({broadcastTo_:$w});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dl(s,e,t){bs(s),t=t||No(e);const n={shape:s,value:e,dtype:t};return K.runKernel(Z0,{},n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dw(s,e,t){const n=H(s,"x","clipByValue");if(R(e<=t,()=>`Error in clip: min (${e}) must be less than or equal to max (${t}).`),e===t)return Dl(n.shape,e,n.dtype);const r={x:n},i={clipValueMin:e,clipValueMax:t};return K.runKernel(M0,r,i)}const An=J({clipByValue_:Dw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ow(s,e,t,n,r="NHWC",i=[1,1],o){const a=H(s,"x","conv2d","float32"),l=H(e,"filter","conv2d","float32");let u=a,c=!1;a.rank===3&&(c=!0,u=ae(a,[1,a.shape[0],a.shape[1],a.shape[2]])),R(u.rank===4,()=>`Error in conv2d: input must be rank 4, but got rank ${u.rank}.`),R(l.rank===4,()=>`Error in conv2d: filter must be rank 4, but got rank ${l.rank}.`),ts("conv2d",n,o);const h=r==="NHWC"?u.shape[3]:u.shape[1];R(h===l.shape[2],()=>`Error in conv2d: depth of input (${h}) must match input depth for filter ${l.shape[2]}.`),R(bi(t,i),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${t} and dilations '${i}'`),R(hi(i),()=>"Error in conv2D: Dilated rates should be larger than 0."),R(hi(t),()=>"Error in conv2D: Strides should be larger than 0.");const d={x:u,filter:l},w={strides:t,pad:n,dataFormat:r,dilations:i,dimRoundingMode:o},k=K.runKernel(L0,d,w);return c?ae(k,[k.shape[1],k.shape[2],k.shape[3]]):k}const Ec=J({conv2d_:Ow});function Mw(s,e,t,n,r="NWC",i=1,o){const a=H(s,"x","conv1d"),l=H(e,"filter","conv1d");let u=a,c=!1;a.rank===2&&(c=!0,u=ae(a,[1,a.shape[0],a.shape[1]])),R(u.rank===3,()=>`Error in conv1d: input must be rank 3, but got rank ${u.rank}.`),R(l.rank===3,()=>`Error in conv1d: filter must be rank 3, but got rank ${l.rank}.`),ts("conv1d",n,o),R(u.shape[2]===l.shape[1],()=>`Error in conv1d: depth of input (${u.shape[2]}) must match input depth for filter ${l.shape[1]}.`),R(bi(t,i),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${t} and dilation '${i}'`),R(hi(i),()=>"Error in conv1D: Dilated rates should be larger than 0."),R(hi(t),()=>"Error in conv1D: Stride should be larger than 0."),R(r==="NWC",()=>`Error in conv1d: got dataFormat of ${r} but only NWC is currently supported.`);const h=ae(l,[1,l.shape[0],l.shape[1],l.shape[2]]),d=ae(u,[u.shape[0],1,u.shape[1],u.shape[2]]),m=Ec(d,h,[1,t],n,"NHWC",[1,i],o);return c?ae(m,[m.shape[2],m.shape[3]]):ae(m,[m.shape[0],m.shape[2],m.shape[3]])}const Pw=J({conv1d_:Mw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rw(s,e,t,n,r,i="NHWC",o){R(s.length===e.rank,()=>`Length of inShape (${s.length}) and rank of dy (${e.rank}) must match`);let a=s,l=e,u=!1;e.rank===3&&(u=!0,l=ae(e,[1,e.shape[0],e.shape[1],e.shape[2]]),a=[1,s[0],s[1],s[2]]),R(a.length===4,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${a.length}.`),R(l.rank===4,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${l.rank}`),R(t.rank===4,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${t.rank}`);const c=i==="NHWC"?a[3]:a[1],h=i==="NHWC"?l.shape[3]:l.shape[1];R(c===t.shape[2],()=>`Error in conv2dDerInput: depth of input (${c}) must match input depth for filter ${t.shape[2]}.`),R(h===t.shape[3],()=>`Error in conv2dDerInput: depth of output (${h}) must match output depth for filter ${t.shape[3]}.`),ts("conv2dDerInput",r,o);const d={dy:l,filter:t},w={strides:n,pad:r,dataFormat:i,dimRoundingMode:o,inputShape:a},k=K.runKernel(F0,d,w);return u?ae(k,[k.shape[1],k.shape[2],k.shape[3]]):k}const Cp=J({conv2DBackpropInput_:Rw});function Lw(s,e,t,n,r,i){const o=H(s,"x","conv2dTranspose"),a=H(e,"filter","conv2dTranspose");return Cp(t,o,a,n,r,"NHWC",i)}const Bw=J({conv2dTranspose_:Lw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fw(s,e,t,n,r="NDHWC",i=[1,1,1]){const o=H(s,"x","conv3d"),a=H(e,"filter","conv3d");let l=o,u=!1;o.rank===4&&(u=!0,l=ae(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),R(l.rank===5,()=>`Error in conv3d: input must be rank 5, but got rank ${l.rank}.`),R(a.rank===5,()=>`Error in conv3d: filter must be rank 5, but got rank ${a.rank}.`),R(l.shape[4]===a.shape[3],()=>`Error in conv3d: depth of input (${l.shape[4]}) must match input depth for filter ${a.shape[3]}.`),R(bi(t,i),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${t} and dilations '${i}'`),R(r==="NDHWC",()=>`Error in conv3d: got dataFormat of ${r} but only NDHWC is currently supported.`),R(hi(i),()=>"Error in conv3D: Dilated rates should be larger than 0."),R(hi(t),()=>"Error in conv3D: Strides should be larger than 0.");const c={x:l,filter:a},h={strides:t,pad:n,dataFormat:r,dilations:i},d=K.runKernel(U0,c,h);return u?ae(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}const Uw=J({conv3d_:Fw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zw(s,e,t,n,r){R(s.length===e.rank,()=>`Length of inShape (${s.length}) and rank of dy (${e.rank}) must match`);let i=s,o=e,a=!1;e.rank===4&&(a=!0,o=ae(e,[1,e.shape[0],e.shape[1],e.shape[2],e.shape[3]]),i=[1,s[0],s[1],s[2],s[3]]);const l=i[4],u=o.shape[4];R(i.length===5,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${i.length}.`),R(o.rank===5,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${o.rank}`),R(t.rank===5,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${t.rank}`),R(l===t.shape[3],()=>`Error in conv3dDerInput: depth of input (${l}) must match input depth for filter ${t.shape[3]}.`),R(u===t.shape[4],()=>`Error in conv3dDerInput: depth of output (${u}) must match output depth for filter ${t.shape[4]}.`);const c={dy:o,filter:t},h={pad:r,strides:n,inputShape:i},d=K.runKernel(z0,c,h);return a?ae(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}const Ww=J({conv3DBackpropInput_:zw});function Gw(s,e,t,n,r){const i=H(s,"x","conv3dTranspose"),o=H(e,"filter","conv3dTranspose");return Ww(t,i,o,n,r)}const Vw=J({conv3dTranspose_:Gw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qw(s,e,t,n,r="NHWC",i=[1,1],o){const a=H(s,"x","depthwiseConv2d","float32"),l=H(e,"filter","depthwiseConv2d","float32");let u=a,c=!1;a.rank===3&&(c=!0,u=ae(a,[1,a.shape[0],a.shape[1],a.shape[2]])),R(u.rank===4,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${u.rank}.`),R(l.rank===4,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${l.rank}.`);const h=r==="NHWC"?u.shape[3]:u.shape[1];R(h===l.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${h}) must match the inChannels dimension in filter ${l.shape[2]}.`),ts("depthwiseConv2d",n,o);const d={x:u,filter:l},w={strides:t,pad:n,dataFormat:r,dilations:i,dimRoundingMode:o},k=K.runKernel(G0,d,w);return c?ae(k,[k.shape[1],k.shape[2],k.shape[3]]):k}const Hw=J({depthwiseConv2d_:qw});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Va(s,e){const t=s.length,n=[];for(let r=0;r<t;r++){const i=t-1-r,o=s[i]||1;(e[e.length-1-r]||1)>1&&o===1&&n.unshift(i)}return n}function jw(s,e){const t=[];for(let n=0;n<e.length;n++){const r=s[s.length-n-1],i=e.length-n-1,o=e[i];(r==null||r===1&&o>1)&&t.unshift(i)}return t}function Ft(s,e){const t=Math.max(s.length,e.length),n=new Array(t);for(let r=0;r<t;r++){let i=s[s.length-r-1];i==null&&(i=1);let o=e[e.length-r-1];if(o==null&&(o=1),i===1)n[t-r-1]=o;else if(o===1)n[t-r-1]=i;else if(i!==o){const a=`Operands could not be broadcast together with shapes ${s} and ${e}.`;throw Error(a)}else n[t-r-1]=i}return n}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kw(s,e){let t=H(s,"a","equal","string_or_numeric"),n=H(e,"b","equal","string_or_numeric");[t,n]=$t(t,n),Ft(t.shape,n.shape);const r={a:t,b:n};return K.runKernel(K0,r)}const kr=J({equal_:Kw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xw(s,e,t){const n=H(e,"a","where"),r=H(t,"b","where"),i=H(s,"condition","where","bool"),o=Ft(Ft(i.shape,n.shape),r.shape),a=ga(i,o),l=ga(n,o),u=ga(r,o),c={condition:a,t:l,e:u};return K.runKernel(Pb,c)}const mr=J({where_:Xw});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yw(s){const t={x:H(s,"x","zerosLike")};return K.runKernel(jb,t)}const Qn=J({zerosLike_:Yw});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zw(s,...e){const t=e.map((r,i)=>H(r,`tensors${i}`,"einsum")),n={equation:s};return K.runKernel(q0,t,n)}const $i=J({einsum_:Zw});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qw(s){const t={x:H(s,"x","elu","float32")};return K.runKernel(H0,t)}const Np=J({elu_:Qw});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jw(s){let e=H(s,"x","erf");R(e.dtype==="int32"||e.dtype==="float32",()=>"Input dtype must be `int32` or `float32`."),e.dtype==="int32"&&(e=De(e,"float32"));const t={x:e};return K.runKernel(j0,t)}const ex=J({erf_:Jw});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $p(s,e){for(let t=0;t<s.length;++t)if(s[s.length-t-1]!==e-1-t)return!1;return!0}function tx(s,e,t){const n=s.length+e.length,r=[];let i=0,o=0;for(let a=0;a<n;a++)t.indexOf(a)===-1?r.push(s[i++]):r.push(e[o++]);return r}function Tc(s,e){const t=[],n=s.length;for(let i=0;i<n;i++)e.indexOf(i)===-1&&t.push(s[i]);const r=e.map(i=>s[i]);return[t,r]}function Dp(s,e){const t=e.map(n=>1);return tx(s,t,e)}function nx(s,e,t){R($p(e,t),()=>`${s} supports only inner-most axes for now. Got axes ${e} and rank-${t} input.`)}function sx(s,e){if($p(s,e))return null;const t=[];for(let n=0;n<e;++n)s.indexOf(n)===-1&&t.push(n);return s.forEach(n=>t.push(n)),t}function rx(s,e){const t=[];for(let n=e-s;n<e;++n)t.push(n);return t}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ix(s,e=null,t=!1){const r={x:H(s,"x","max")},i={reductionIndices:e,keepDims:t};return K.runKernel(hb,r,i)}const Ds=J({max_:ix});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ox(s,e=null,t=!1){const r={x:H(s,"x","min")},i={axis:e,keepDims:t};return K.runKernel(mb,r,i)}const Yh=J({min_:ox});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ax(s,e){let t=H(s,"base","pow"),n=H(e,"exp","pow");[t,n]=$t(t,n);const r={a:t,b:n};return K.runKernel(Eb,r)}const qa=J({pow_:ax});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qt(s,e){if((ln(s)&&e!=="string"||Array.isArray(s))&&e!=="complex64")throw new Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if(e==="string"&&ln(s)&&!(s instanceof Uint8Array))throw new Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return $l(s,[],[],e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lx(s){const t={x:H(s,"x","sqrt","float32")};return K.runKernel(Fb,t)}const Cn=J({sqrt_:lx});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ux(s){const e=H(s,"x","square"),t={};return K.runKernel("Square",{x:e},t)}const Os=J({square_:ux});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cx(s,e=null,t=!1){let n=H(s,"x","sum");n.dtype==="bool"&&(n=De(n,"int32"));const r={x:n},i={axis:e,keepDims:t};return K.runKernel(Ub,r,i)}const Ae=J({sum_:cx});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hx(s,e="euclidean",t=null,n=!1){s=H(s,"x","norm");const r=Op(s,e,t);let i=r.shape;if(n){const o=Co(t,s.shape);i=Dp(r.shape,o)}return ae(r,i)}function Op(s,e,t=null){if(s.rank===0)return Tt(s);if(s.rank!==1&&t===null)return Op(ae(s,[-1]),e,t);if(s.rank===1||typeof t=="number"||Array.isArray(t)&&t.length===1){if(e===1)return Ae(Tt(s),t);if(e===1/0)return Ds(Tt(s),t);if(e===-1/0)return Yh(Tt(s),t);if(e==="euclidean"||e===2)return Cn(Ae(qa(Tt(s),Qt(2,"int32")),t));throw new Error(`Error in norm: invalid ord value: ${e}`)}if(Array.isArray(t)&&t.length===2){if(e===1)return Ds(Ae(Tt(s),t[0]),t[1]-1);if(e===1/0)return Ds(Ae(Tt(s),t[1]),t[0]);if(e===-1/0)return Yh(Ae(Tt(s),t[1]),t[0]);if(e==="fro"||e==="euclidean")return Cn(Ae(Os(s),t));throw new Error(`Error in norm: invalid ord value: ${e}`)}throw new Error(`Error in norm: invalid axis: ${t}`)}const Mp=J({norm_:hx});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fx(s){const t={x:H(s,"x","exp")};return K.runKernel(X0,t)}const Fu=J({exp_:fx});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dx(s,e=0){const t=H(s,"x","expandDims","string_or_numeric");R(e<=t.rank,()=>"Axis must be <= rank of the tensor");const n={input:t},r={dim:e};return K.runKernel(Y0,n,r)}const jn=J({expandDims_:dx});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function px(s,e){const t=H(s,"x","tile","string_or_numeric");R(t.rank===e.length,()=>`Error in transpose: rank of input ${t.rank} must match length of reps ${e}.`);const n={x:t},r={reps:e};return K.runKernel(op,n,r)}const ya=J({tile_:px});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mx(s,e,t,n="float32"){e==null&&(e=s);const r=ct([s,e],n),i=s<=e?s:e;for(let a=0;a<i;++a)r.set(1,a,a);const o=ae(r.toTensor(),[s,e]);if(t==null)return o;if(t.length===1)return ya(jn(o,0),[t[0],1,1]);if(t.length===2)return ya(jn(jn(o,0),0),[t[0],t[1],1,1]);if(t.length===3)return ya(jn(jn(jn(o,0),0),0),[t[0],t[1],t[2],1,1]);throw new Error(`eye() currently supports only 1D and 2D batchShapes, but received ${t.length}D.`)}const Pp=J({eye_:mx});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gx(s){const t={x:H(s,"x","floor","float32")};return K.runKernel(J0,t)}const yx=J({floor_:gx});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bx(s,e,t=0,n=0){const r=H(s,"x","gather"),i=H(e,"indices","gather","int32"),o={x:r,indices:i},a={axis:t,batchDims:n};return K.runKernel(tb,o,a)}const wx=J({gather_:bx});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xx(s,e){let t=H(s,"a","greater","string_or_numeric"),n=H(e,"b","greater","string_or_numeric");[t,n]=$t(t,n),Ft(t.shape,n.shape);const r={a:t,b:n};return K.runKernel(nb,r)}const $o=J({greater_:xx});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _x(s,e){let t=H(s,"a","greaterEqual","string_or_numeric"),n=H(e,"b","greaterEqual","string_or_numeric");[t,n]=$t(t,n),Ft(t.shape,n.shape);const r={a:t,b:n};return K.runKernel(sb,r)}const vx=J({greaterEqual_:_x});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sx(s){const t={input:H(s,"input","imag")};return K.runKernel(rb,t)}const kx=J({imag_:Sx});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ix(s,e=.2){const n={x:H(s,"x","leakyRelu")},r={alpha:e};return K.runKernel(ib,n,r)}const Ex=J({leakyRelu_:Ix});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tx(s,e){let t=H(s,"a","less","string_or_numeric"),n=H(e,"b","less","string_or_numeric");[t,n]=$t(t,n),Ft(t.shape,n.shape);const r={a:t,b:n};return K.runKernel(ob,r)}const Zh=J({less_:Tx});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ax(s,e){let t=H(s,"a","lessEqual","string_or_numeric"),n=H(e,"b","lessEqual","string_or_numeric");[t,n]=$t(t,n),Ft(t.shape,n.shape);const r={a:t,b:n};return K.runKernel(ab,r)}const Rp=J({lessEqual_:Ax});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cx(s){const t={x:H(s,"x","log","float32")};return K.runKernel(lb,t)}const Ir=J({log_:Cx});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nx(s){const t={x:H(s,"x","log1p")};return K.runKernel(ub,t)}const $x=J({log1p_:Nx});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dx(s,e){R(Eu(s),()=>"The f passed in variableGrads(f) must be a function"),R(e==null||Array.isArray(e)&&e.every(u=>u instanceof za),()=>"The varList passed in variableGrads(f, varList) must be an array of variables");const t=e!=null;if(!t){e=[];for(const u in K.registeredVariables)e.push(K.registeredVariables[u])}const n=t?e.filter(u=>!u.trainable):null,r=e.length;e=e.filter(u=>u.trainable),R(e.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${r} variables is trainable.`);const i=!0,{value:o,grads:a}=K.gradients(s,e,null,i);R(a.some(u=>u!=null),()=>"Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize()."),R(o.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${o.rank} tensor`);const l={};return e.forEach((u,c)=>{a[c]!=null&&(l[u.name]=a[c])}),n?.forEach(u=>l[u.name]=null),{value:o,grads:l}}function Uu(s){return K.customGrad(s)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ox(s){const t={x:H(s,"x","neg")};return K.runKernel(wb,t)}const wi=J({neg_:Ox});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mx(s){const t={x:H(s,"x","softplus")};return K.runKernel(Bb,t)}const Ac=J({softplus_:Mx});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Px(s,e){let t=H(s,"a","sub"),n=H(e,"b","sub");[t,n]=$t(t,n);const r={a:t,b:n};return K.runKernel(Gb,r)}const ke=J({sub_:Px});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rx(s,e=-1){const t=H(s,"logits","logSoftmax");if(e===-1&&(e=t.rank-1),e!==t.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${t.rank} and axis was ${e}`);return Uu((r,i)=>{const a=Ds(r,e,!0),l=ke(r,a),u=ke(De(l,"float32"),Ir(Ae(Fu(l),e,!0)));return i([u]),{value:u,gradFunc:(h,d)=>{const[w]=d,k=!0,A=Fu(w);return ke(h,ne(Ae(h,e,k),A))}}})(t)}const Lx=J({logSoftmax_:Rx});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bx(s,e){const t=H(s,"a","logicalAnd","bool"),n=H(e,"b","logicalAnd","bool");Ft(t.shape,n.shape);const r={a:t,b:n};return K.runKernel(cb,r)}const Ol=J({logicalAnd_:Bx});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fx(s,e,t,n,r){const i=H(s,"x","maxPool"),o=1;let a=i,l=!1;i.rank===3&&(l=!0,a=ae(i,[1,i.shape[0],i.shape[1],i.shape[2]])),R(a.rank===4,()=>`Error in maxPool: input must be rank 4 but got rank ${a.rank}.`),R(bi(t,o),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${t} and dilations '${o}'`),ts("maxPool",n,r);const u={x:a},c={filterSize:e,strides:t,pad:n,dimRoundingMode:r},h=K.runKernel(np,u,c);return l?ae(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const Ux=J({maxPool_:Fx});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zx(s,e=[1,1,1],t,n,r,i="NDHWC"){const o=H(s,"x","maxPool3d");let a=o,l=!1;o.rank===4&&(l=!0,a=ae(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),R(a.rank===5,()=>`Error in maxPool3d: x must be rank 5 but got rank ${a.rank}.`),R(i==="NDHWC",()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${i}`),ts("maxPool3d",n,r);const u={x:a},c={filterSize:e,strides:t,pad:n,dimRoundingMode:r,dataFormat:i},h=K.runKernel(db,u,c);return l?ae(h,[h.shape[1],h.shape[2],h.shape[3],h.shape[4]]):h}const Wx=J({maxPool3d_:zx});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gx(s,e){let t=H(s,"a","maximum"),n=H(e,"b","maximum");[t,n]=$t(t,n),t.dtype==="bool"&&(t=De(t,"int32"),n=De(n,"int32")),Ft(t.shape,n.shape);const r={a:t,b:n};return K.runKernel(fb,r)}const xi=J({maximum_:Gx});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vx(s,e=null,t=!1){const r={x:H(s,"x","mean")},i={axis:e,keepDims:t};return K.runKernel(pb,r,i)}const ut=J({mean_:Vx});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fi(s,e="float32"){if(bs(s),e==="complex64"){const n=fi(s,"float32"),r=fi(s,"float32");return _c(n,r)}const t=Rs(me(s),e);return K.makeTensor(t,s,e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cc(s,e="float32"){if(bs(s),e==="complex64"){const n=Cc(s,"float32"),r=fi(s,"float32");return _c(n,r)}const t=Yd(me(s),e);return K.makeTensor(t,s,e)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qx(s,e){let t=H(s,"a","minimum"),n=H(e,"b","minimum");[t,n]=$t(t,n),t.dtype==="bool"&&(t=De(t,"int32"),n=De(n,"int32")),Ft(t.shape,n.shape);const r={a:t,b:n};return K.runKernel(gb,r)}const Ha=J({minimum_:qx});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hx(s,e,t=1,n=0,r="int32"){if(e<2)throw new Error(`Error in oneHot: depth must be >=2, but it is ${e}`);const o={indices:H(s,"indices","oneHot","int32")},a={dtype:r,depth:e,onValue:t,offValue:n};return K.runKernel(kb,o,a)}const jx=J({oneHot_:Hx});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kx(s){const t={x:H(s,"x","onesLike")};return K.runKernel(Sb,t)}const Lp=J({onesLike_:Kx});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xx(s,e){const t=H(s,"x","prelu"),n=H(e,"alpha","prelu"),r={x:t,alpha:n};return K.runKernel(Tb,r)}const Yx=J({prelu_:Xx});var ba={exports:{}},Zx=ba.exports,Qh;function Qx(){return Qh||(Qh=1,function(s){(function(e,t,n){function r(l){var u=this,c=a();u.next=function(){var h=2091639*u.s0+u.c*23283064365386963e-26;return u.s0=u.s1,u.s1=u.s2,u.s2=h-(u.c=h|0)},u.c=1,u.s0=c(" "),u.s1=c(" "),u.s2=c(" "),u.s0-=c(l),u.s0<0&&(u.s0+=1),u.s1-=c(l),u.s1<0&&(u.s1+=1),u.s2-=c(l),u.s2<0&&(u.s2+=1),c=null}function i(l,u){return u.c=l.c,u.s0=l.s0,u.s1=l.s1,u.s2=l.s2,u}function o(l,u){var c=new r(l),h=u&&u.state,d=c.next;return d.int32=function(){return c.next()*4294967296|0},d.double=function(){return d()+(d()*2097152|0)*11102230246251565e-32},d.quick=d,h&&(typeof h=="object"&&i(h,c),d.state=function(){return i(c,{})}),d}function a(){var l=4022871197,u=function(c){c=String(c);for(var h=0;h<c.length;h++){l+=c.charCodeAt(h);var d=.02519603282416938*l;l=d>>>0,d-=l,d*=l,l=d>>>0,d-=l,l+=d*4294967296}return(l>>>0)*23283064365386963e-26};return u}t&&t.exports?t.exports=o:this.alea=o})(Zx,s)}(ba)),ba.exports}var wa={exports:{}},Jx=wa.exports,Jh;function e2(){return Jh||(Jh=1,function(s){(function(e,t,n){function r(a){var l=this,u="";l.x=0,l.y=0,l.z=0,l.w=0,l.next=function(){var h=l.x^l.x<<11;return l.x=l.y,l.y=l.z,l.z=l.w,l.w^=l.w>>>19^h^h>>>8},a===(a|0)?l.x=a:u+=a;for(var c=0;c<u.length+64;c++)l.x^=u.charCodeAt(c)|0,l.next()}function i(a,l){return l.x=a.x,l.y=a.y,l.z=a.z,l.w=a.w,l}function o(a,l){var u=new r(a),c=l&&l.state,h=function(){return(u.next()>>>0)/4294967296};return h.double=function(){do var d=u.next()>>>11,w=(u.next()>>>0)/4294967296,k=(d+w)/(1<<21);while(k===0);return k},h.int32=u.next,h.quick=h,c&&(typeof c=="object"&&i(c,u),h.state=function(){return i(u,{})}),h}t&&t.exports?t.exports=o:this.xor128=o})(Jx,s)}(wa)),wa.exports}var xa={exports:{}},t2=xa.exports,ef;function n2(){return ef||(ef=1,function(s){(function(e,t,n){function r(a){var l=this,u="";l.next=function(){var h=l.x^l.x>>>2;return l.x=l.y,l.y=l.z,l.z=l.w,l.w=l.v,(l.d=l.d+362437|0)+(l.v=l.v^l.v<<4^(h^h<<1))|0},l.x=0,l.y=0,l.z=0,l.w=0,l.v=0,a===(a|0)?l.x=a:u+=a;for(var c=0;c<u.length+64;c++)l.x^=u.charCodeAt(c)|0,c==u.length&&(l.d=l.x<<10^l.x>>>4),l.next()}function i(a,l){return l.x=a.x,l.y=a.y,l.z=a.z,l.w=a.w,l.v=a.v,l.d=a.d,l}function o(a,l){var u=new r(a),c=l&&l.state,h=function(){return(u.next()>>>0)/4294967296};return h.double=function(){do var d=u.next()>>>11,w=(u.next()>>>0)/4294967296,k=(d+w)/(1<<21);while(k===0);return k},h.int32=u.next,h.quick=h,c&&(typeof c=="object"&&i(c,u),h.state=function(){return i(u,{})}),h}t&&t.exports?t.exports=o:this.xorwow=o})(t2,s)}(xa)),xa.exports}var _a={exports:{}},s2=_a.exports,tf;function r2(){return tf||(tf=1,function(s){(function(e,t,n){function r(a){var l=this;l.next=function(){var c=l.x,h=l.i,d,w;return d=c[h],d^=d>>>7,w=d^d<<24,d=c[h+1&7],w^=d^d>>>10,d=c[h+3&7],w^=d^d>>>3,d=c[h+4&7],w^=d^d<<7,d=c[h+7&7],d=d^d<<13,w^=d^d<<9,c[h]=w,l.i=h+1&7,w};function u(c,h){var d,w=[];if(h===(h|0))w[0]=h;else for(h=""+h,d=0;d<h.length;++d)w[d&7]=w[d&7]<<15^h.charCodeAt(d)+w[d+1&7]<<13;for(;w.length<8;)w.push(0);for(d=0;d<8&&w[d]===0;++d);for(d==8?w[7]=-1:w[d],c.x=w,c.i=0,d=256;d>0;--d)c.next()}u(l,a)}function i(a,l){return l.x=a.x.slice(),l.i=a.i,l}function o(a,l){a==null&&(a=+new Date);var u=new r(a),c=l&&l.state,h=function(){return(u.next()>>>0)/4294967296};return h.double=function(){do var d=u.next()>>>11,w=(u.next()>>>0)/4294967296,k=(d+w)/(1<<21);while(k===0);return k},h.int32=u.next,h.quick=h,c&&(c.x&&i(c,u),h.state=function(){return i(u,{})}),h}t&&t.exports?t.exports=o:this.xorshift7=o})(s2,s)}(_a)),_a.exports}var va={exports:{}},i2=va.exports,nf;function o2(){return nf||(nf=1,function(s){(function(e,t,n){function r(a){var l=this;l.next=function(){var c=l.w,h=l.X,d=l.i,w,k;return l.w=c=c+1640531527|0,k=h[d+34&127],w=h[d=d+1&127],k^=k<<13,w^=w<<17,k^=k>>>15,w^=w>>>12,k=h[d]=k^w,l.i=d,k+(c^c>>>16)|0};function u(c,h){var d,w,k,A,m,S=[],b=128;for(h===(h|0)?(w=h,h=null):(h=h+"\0",w=0,b=Math.max(b,h.length)),k=0,A=-32;A<b;++A)h&&(w^=h.charCodeAt((A+32)%h.length)),A===0&&(m=w),w^=w<<10,w^=w>>>15,w^=w<<4,w^=w>>>13,A>=0&&(m=m+1640531527|0,d=S[A&127]^=w+m,k=d==0?k+1:0);for(k>=128&&(S[(h&&h.length||0)&127]=-1),k=127,A=512;A>0;--A)w=S[k+34&127],d=S[k=k+1&127],w^=w<<13,d^=d<<17,w^=w>>>15,d^=d>>>12,S[k]=w^d;c.w=m,c.X=S,c.i=k}u(l,a)}function i(a,l){return l.i=a.i,l.w=a.w,l.X=a.X.slice(),l}function o(a,l){a==null&&(a=+new Date);var u=new r(a),c=l&&l.state,h=function(){return(u.next()>>>0)/4294967296};return h.double=function(){do var d=u.next()>>>11,w=(u.next()>>>0)/4294967296,k=(d+w)/(1<<21);while(k===0);return k},h.int32=u.next,h.quick=h,c&&(c.X&&i(c,u),h.state=function(){return i(u,{})}),h}t&&t.exports?t.exports=o:this.xor4096=o})(i2,s)}(va)),va.exports}var Sa={exports:{}},a2=Sa.exports,sf;function l2(){return sf||(sf=1,function(s){(function(e,t,n){function r(a){var l=this,u="";l.next=function(){var h=l.b,d=l.c,w=l.d,k=l.a;return h=h<<25^h>>>7^d,d=d-w|0,w=w<<24^w>>>8^k,k=k-h|0,l.b=h=h<<20^h>>>12^d,l.c=d=d-w|0,l.d=w<<16^d>>>16^k,l.a=k-h|0},l.a=0,l.b=0,l.c=-1640531527,l.d=1367130551,a===Math.floor(a)?(l.a=a/4294967296|0,l.b=a|0):u+=a;for(var c=0;c<u.length+20;c++)l.b^=u.charCodeAt(c)|0,l.next()}function i(a,l){return l.a=a.a,l.b=a.b,l.c=a.c,l.d=a.d,l}function o(a,l){var u=new r(a),c=l&&l.state,h=function(){return(u.next()>>>0)/4294967296};return h.double=function(){do var d=u.next()>>>11,w=(u.next()>>>0)/4294967296,k=(d+w)/(1<<21);while(k===0);return k},h.int32=u.next,h.quick=h,c&&(typeof c=="object"&&i(c,u),h.state=function(){return i(u,{})}),h}t&&t.exports?t.exports=o:this.tychei=o})(a2,s)}(Sa)),Sa.exports}var ka={exports:{}},u2={},c2=Object.freeze({__proto__:null,default:u2}),h2=Jb(c2),f2=ka.exports,rf;function d2(){return rf||(rf=1,function(s){(function(e,t,n){var r=256,i=6,o=52,a="random",l=n.pow(r,i),u=n.pow(2,o),c=u*2,h=r-1,d;function w(v,_,E){var D=[];_=_==!0?{entropy:!0}:_||{};var M=S(m(_.entropy?[v,f(t)]:v??b(),3),D),$=new k(D),C=function(){for(var g=$.g(i),p=l,y=0;g<u;)g=(g+y)*r,p*=r,y=$.g(1);for(;g>=c;)g/=2,p/=2,y>>>=1;return(g+y)/p};return C.int32=function(){return $.g(4)|0},C.quick=function(){return $.g(4)/4294967296},C.double=C,S(f($.S),t),(_.pass||E||function(g,p,y,x){return x&&(x.S&&A(x,$),g.state=function(){return A($,{})}),y?(n[a]=g,p):g})(C,M,"global"in _?_.global:this==n,_.state)}function k(v){var _,E=v.length,D=this,M=0,$=D.i=D.j=0,C=D.S=[];for(E||(v=[E++]);M<r;)C[M]=M++;for(M=0;M<r;M++)C[M]=C[$=h&$+v[M%E]+(_=C[M])],C[$]=_;(D.g=function(g){for(var p,y=0,x=D.i,I=D.j,N=D.S;g--;)p=N[x=h&x+1],y=y*r+N[h&(N[x]=N[I=h&I+p])+(N[I]=p)];return D.i=x,D.j=I,y})(r)}function A(v,_){return _.i=v.i,_.j=v.j,_.S=v.S.slice(),_}function m(v,_){var E=[],D=typeof v,M;if(_&&D=="object")for(M in v)try{E.push(m(v[M],_-1))}catch{}return E.length?E:D=="string"?v:v+"\0"}function S(v,_){for(var E=v+"",D,M=0;M<E.length;)_[h&M]=h&(D^=_[h&M]*19)+E.charCodeAt(M++);return f(_)}function b(){try{var v;return d&&(v=d.randomBytes)?v=v(r):(v=new Uint8Array(r),(e.crypto||e.msCrypto).getRandomValues(v)),f(v)}catch{var _=e.navigator,E=_&&_.plugins;return[+new Date,e,E,e.screen,f(t)]}}function f(v){return String.fromCharCode.apply(0,v)}if(S(n.random(),t),s.exports){s.exports=w;try{d=h2}catch{}}else n["seed"+a]=w})(typeof self<"u"?self:f2,[],Math)}(ka)),ka.exports}var nu,of;function p2(){if(of)return nu;of=1;var s=Qx(),e=e2(),t=n2(),n=r2(),r=o2(),i=l2(),o=d2();return o.alea=s,o.xor128=e,o.xorwow=t,o.xorshift7=n,o.xor4096=r,o.tychei=i,nu=o,nu}var Bp=p2();/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Fp{constructor(e,t,n,r,i){this.mean=e,this.stdDev=t,this.dtype=n,this.nextVal=NaN,this.truncated=r,this.truncated&&(this.upper=this.mean+this.stdDev*2,this.lower=this.mean-this.stdDev*2);const o=i||Math.random();this.random=Bp.alea(o.toString())}nextValue(){if(!isNaN(this.nextVal)){const r=this.nextVal;return this.nextVal=NaN,r}let e,t,n=!1;for(;!n;){let r,i,o;do r=2*this.random()-1,i=2*this.random()-1,o=r*r+i*i;while(o>=1||o===0);const a=Math.sqrt(-2*Math.log(o)/o);e=this.mean+this.stdDev*r*a,t=this.mean+this.stdDev*i*a,(!this.truncated||this.isValidTruncated(e))&&(n=!0)}return(!this.truncated||this.isValidTruncated(t))&&(this.nextVal=this.convertValue(t)),this.convertValue(e)}convertValue(e){return this.dtype==null||this.dtype==="float32"?e:Math.round(e)}isValidTruncated(e){return e<=this.upper&&e>=this.lower}}class m2{constructor(e=0,t=1,n,r){if(this.canReturnFloat=()=>this.dtype==null||this.dtype==="float32",this.min=e,this.range=t-e,this.dtype=n,r==null&&(r=Math.random()),typeof r=="number"&&(r=r.toString()),!this.canReturnFloat()&&this.range<=1)throw new Error(`The difference between ${e} - ${t} <= 1 and dtype is not float`);this.random=Bp.alea(r)}convertValue(e){return this.canReturnFloat()?e:Math.round(e)}nextValue(){return this.convertValue(this.min+this.range*this.random())}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function g2(s,e=0,t=1,n,r){if(bs(s),n!=null&&n==="bool")throw new Error(`Unsupported data type ${n}`);const i=new Fp(e,t,n,!1,r),o=ct(s,n);for(let a=0;a<o.values.length;a++)o.values[a]=i.nextValue();return o.toTensor()}const y2=J({randomNormal_:g2});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function b2(s,e=0,t=1,n="float32",r){bs(s);const i=ct(s,n),o=new m2(e,t,null,r);for(let a=0;a<i.values.length;a++)i.values[a]=o.nextValue();return i.toTensor()}const Up=J({randomUniform_:b2});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ja(s,e,t=1,n="float32"){if(t===0)throw new Error("Cannot have a step of zero");const r={start:s,stop:e,step:t,dtype:n};return K.runKernel(Ab,{},r)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function w2(s){const t={input:H(s,"input","real")};return K.runKernel(Cb,t)}const x2=J({real_:w2});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _2(s){const t={x:H(s,"x","relu")};return K.runKernel(Nb,t)}const Do=J({relu_:_2});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function v2(s){const t={x:H(s,"x","relu6")};return K.runKernel(Ob,t)}const S2=J({relu6_:v2});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function k2(s){const t={x:H(s,"x","round")};return K.runKernel(Mb,t)}const I2=J({round_:k2});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function E2(s){const t={x:H(s,"x","selu")};return K.runKernel(Rb,t)}const T2=J({selu_:E2});function A2(s,e,t,n,r,i=[1,1],o="NHWC"){const a=H(s,"x","separableConv2d"),l=H(e,"depthwiseFilter","separableConv2d"),u=H(t,"pointwiseFilter","separableConv2d");let c=a,h=!1;if(a.rank===3&&(h=!0,c=ae(a,[1,a.shape[0],a.shape[1],a.shape[2]])),o==="NCHW")throw new Error("separableConv2d currently does not support dataFormat NCHW; only NHWC is supported");R(c.rank===4,()=>`Error in separableConv2d: input must be rank 4, but got rank ${c.rank}.`),R(l.rank===4,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${l.rank}.`),R(u.rank===4,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${l.rank}.`),R(u.shape[0]===1,()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${u.shape[0]}.`),R(u.shape[1]===1,()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${u.shape[1]}.`);const d=l.shape[2],w=l.shape[3];R(u.shape[2]===d*w,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${d*w}, but got ${u.shape[2]}.`);const k=Hw(c,l,n,r,o,i),m=Ec(k,u,1,"valid",o);return h?ae(m,[m.shape[1],m.shape[2],m.shape[3]]):m}const C2=J({separableConv2d_:A2});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function N2(s,e,t){const n=H(s,"x","slice1d");return R(n.rank===1,()=>`slice1d expects a rank-1 tensor, but got a rank-${n.rank} tensor`),dt(n,[e],[t])}const Nc=J({slice1d_:N2});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $2(s,e,t){const n=H(s,"x","slice2d");return R(n.rank===2,()=>`slice2d expects a rank-2 tensor, but got a rank-${n.rank} tensor`),dt(n,e,t)}const zp=J({slice2d_:$2});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function D2(s,e,t){const n=H(s,"x","slice3d");return R(n.rank===3,()=>`slice3d expects a rank-3 tensor, but got a rank-${n.rank} tensor`),dt(n,e,t)}const $c=J({slice3d_:D2});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function O2(s,e=-1){const t=H(s,"logits","softmax","float32");if(e===-1&&(e=t.rank-1),e!==t.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${t.rank} and dim was ${e}`);const n={logits:t},r={dim:e};return K.runKernel(Wb,n,r)}const Wp=J({softmax_:O2});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function M2(s,e,t=0){const r={x:H(s,"x","split")},i={numOrSizeSplits:e,axis:t};return K.runKernel(zb,r,i)}const Gp=J({split_:M2});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function P2(s,e){const t=H(s,"x","squeeze","string_or_numeric");return ae(t,g0(t.shape,e).newShape)}const Ml=J({squeeze_:P2});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function R2(s,e=0){const t=_p(s,"tensors","stack","string_or_numeric");R(t.length>=1,()=>"Pass at least one tensor to tf.stack"),t.length>0&&R(e<=t[0].rank,()=>"Axis must be <= rank of the tensor");const n=t,r={axis:e};return K.runKernel(Ib,n,r)}const Ka=J({stack_:R2});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function L2(s,e=0){const n={x:H(s,"x","step")},r={alpha:e};return K.runKernel(Kb,n,r)}const B2=J({step_:L2});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function su(s,e,t){if(Kd(s),e!=null&&e.length!==2)throw new Error("tensor2d() requires shape to have two numbers");const n=Nl(s,t);if(n.length!==2&&n.length!==1)throw new Error("tensor2d() requires values to be number[][] or flat/TypedArray");if(n.length===1&&e==null)throw new Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return $l(s,e,n,t)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function F2(s,e=0,t=1,n,r){if(bs(s),n!=null&&n==="bool")throw new Error("Unsupported data type $ { dtype }");const i=new Fp(e,t,n,!0,r),o=ct(s,n);for(let a=0;a<o.values.length;a++)o.values[a]=i.nextValue();return o.toTensor()}const Vp=J({truncatedNormal_:F2});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function U2(s,e=0){const t=H(s,"x","unstack","string_or_numeric");R(e>=-t.shape.length&&e<t.shape.length,()=>`Axis = ${e} is not in [-${t.shape.length}, ${t.shape.length})`);const n={value:t},r={axis:e};return K.runKernel(Hb,n,r)}const qp=J({unstack_:U2});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function z2(s,e=!0,t,n){return K.makeVariable(s,e,t,n)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function W2(s,e,t){const n=H(s,"x","transpose");if(e==null&&(e=n.shape.map((o,a)=>a).reverse()),R(n.rank===e.length,()=>`Error in transpose: rank of input ${n.rank} must match length of perm ${e}.`),e.forEach(o=>{R(o>=0&&o<n.rank,()=>`All entries in 'perm' must be between 0 and ${n.rank-1} but got ${e}`)}),n.rank<=1)return n.clone();const r={x:n},i={perm:e};return n.dtype==="complex64"?Q(()=>{let o=x2(n),a=kx(n);return o=K.runKernel(Yl,{x:o},i),a=K.runKernel(Yl,{x:a},i),t&&(a=wi(a)),_c(o,a)}):K.runKernel(Yl,r,i)}const He=J({transpose_:W2});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function G2(s,e,t,n,r,i="NHWC",o){let a=s;s.rank===3&&(a=ae(s,[1,s.shape[0],s.shape[1],s.shape[2]]));let l=e;l.rank===3&&(l=ae(e,[1,e.shape[0],e.shape[1],e.shape[2]])),R(a.rank===4,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${a.shape}.`),R(l.rank===4,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${l.shape}.`),R(t.length===4,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${t}.`);const u=i==="NHWC"?a.shape[3]:a.shape[1],c=i==="NHWC"?l.shape[3]:l.shape[1];R(u===t[2],()=>`Error in conv2dDerFilter: depth of input ${u}) must match input depth in filter (${t[2]}.`),R(c===t[3],()=>`Error in conv2dDerFilter: depth of dy (${c}) must match output depth for filter (${t[3]}).`),ts("conv2dDerFilter",r,o);const h={x:a,dy:l},d={strides:n,pad:r,dataFormat:i,dimRoundingMode:o,filterShape:t};return K.runKernel(B0,h,d)}const V2=J({conv2DBackpropFilter_:G2});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function q2(s,e,t){if(t==null||t==="linear")return s;if(t==="relu")return ne(s,B2(e));throw new Error(`Cannot compute gradient for fused activation ${t}.`)}function H2(s,e){let t=e;const n=jw(s.shape,e.shape);return n.length>0&&(t=Ae(t,n)),ae(t,s.shape)}function j2(s,e,t,n){if(e==="linear")return s;if(e==="relu")return Do(s);if(e==="elu")return Np(s);if(e==="relu6")return S2(s);if(e==="prelu")return Yx(s,t);if(e==="leakyrelu")return Ex(s,n);if(e==="sigmoid")return kc(s);throw new Error(`Unknown fused activation ${e}.`)}const K2=(s,e)=>!(s>0)||e==="linear";/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function X2({x:s,filter:e,strides:t,pad:n,dataFormat:r="NHWC",dilations:i=[1,1],dimRoundingMode:o,bias:a,activation:l="linear",preluActivationWeights:u,leakyreluAlpha:c}){if(l=l||"linear",K2(K.state.gradientDepth,l)===!1){R(r==="NHWC",()=>`Error in fused conv2d: got dataFormat of ${r} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let E=Ec(s,e,t,n,r,i,o);return a!=null&&(E=he(E,a)),j2(E,l,u,c)}const h=H(s,"x","conv2d","float32"),d=H(e,"filter","conv2d","float32");let w=h,k=!1;h.rank===3&&(k=!0,w=ae(h,[1,h.shape[0],h.shape[1],h.shape[2]])),R(w.rank===4,()=>`Error in fused conv2d: input must be rank 4, but got rank ${w.rank}.`),R(d.rank===4,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${d.rank}.`),ts("fused conv2d",n,o);const A=r==="NHWC"?w.shape[3]:w.shape[1];R(d.shape[2]===A,()=>`Error in conv2d: depth of input (${A}) must match input depth for filter ${d.shape[2]}.`),R(bi(t,i),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${t} and dilations '${i}'`);const m=Sc(w.shape,d.shape,t,i,n,o);let S;a!=null&&(S=H(a,"bias","fused conv2d"),[S]=$t(S,h),r==="NHWC"?Ft(m.outShape,S.shape):(R(S.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${S.shape.length}.`),R(S.shape.length===0||S.shape[0]===m.outChannels||S.shape[0]===1,()=>`Error in fused conv2d: bias shape (${S.shape}) is not compatible with the number of output channels (${m.outChannels})`)));let b;if(u!=null){const E=u.shape;if(R(E.length<=1||E.length===3,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${E.length}.`),E.length===1)R(E[0]===1||E[0]===m.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${E}) is not compatible with the number of output channels (${m.outChannels}).`);else if(E.length===3)try{Ft(E,m.outShape)}catch{const M=`Error in fused conv2d: PReLU activation weights (${E}) is not compatible with the output shape of the conv2d (${m.outShape}).`;throw Error(M)}b=H(u,"prelu weights","fused conv2d")}const f=(E,D)=>{R(r==="NHWC",()=>`Error in gradient of fused conv2D: got dataFormat of ${r} but only NHWC is currently supported.`);const[M,$,C,g]=D,p=q2(E,C,l);R(Bu(i),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${i}'`);const y=Cp($.shape,p,M,t,n),x=V2($,p,M.shape,t,n),I=[y,x];if(g!=null){const N=H2(g,p);I.push(N)}return I},v={x:w,filter:d,bias:S,preluActivationWeights:b},_={strides:t,pad:n,dataFormat:r,dilations:i,dimRoundingMode:o,activation:l,leakyreluAlpha:c};return a==null?Uu((D,M,$)=>{let C=K.runKernel(Au,v,_);return $([M,D,C]),k&&(C=ae(C,[C.shape[1],C.shape[2],C.shape[3]])),{value:C,gradFunc:f}})(w,d):Uu((D,M,$,C)=>{let g=K.runKernel(Au,v,_);return C([M,D,g,$]),k&&(g=ae(g,[g.shape[1],g.shape[2],g.shape[3]])),{value:g,gradFunc:f}})(w,d,S)}const Y2=J({fusedConv2d_:X2});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Z2(s,e,t,n,r="bilinear",i=0){const o=H(s,"image","cropAndResize"),a=H(e,"boxes","cropAndResize","float32"),l=H(t,"boxInd","cropAndResize","int32"),u=a.shape[0];R(o.rank===4,()=>`Error in cropAndResize: image must be rank 4,but got rank ${o.rank}.`),R(a.rank===2&&a.shape[1]===4,()=>`Error in cropAndResize: boxes must be have size [${u},4] but had shape ${a.shape}.`),R(l.rank===1&&l.shape[0]===u,()=>`Error in cropAndResize: boxInd must be have size [${u}] but had shape ${a.shape}.`),R(n.length===2,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${n.length}.`),R(n[0]>=1&&n[1]>=1,()=>`cropSize must be atleast [1,1], but was ${n}`),R(r==="bilinear"||r==="nearest",()=>`method must be bilinear or nearest, but was ${r}`);const c={image:o,boxes:a,boxInd:l},h={method:r,extrapolationValue:i,cropSize:n};return K.runKernel(W0,c,h)}const Q2=J({cropAndResize_:Z2});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function J2(s){const e=H(s,"image","flipLeftRight","float32");R(e.rank===4,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${e.rank}.`);const t={image:e};return K.runKernel(Q0,t,{})}const e_=J({flipLeftRight_:J2});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function t_(s){const e=H(s,"image","grayscaleToRGB"),t=e.rank-1,n=e.shape[t];R(e.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${e.rank}.`),R(n===1,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${n}.`);const r=new Array(e.rank);return r.fill(1,0,t),r[t]=3,ya(e,r)}const n_=J({grayscaleToRGB_:t_});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function s_(s){const e=H(s,"image","RGBToGrayscale"),t=e.rank-1,n=e.shape[t];R(e.rank>=2,()=>`Error in RGBToGrayscale: images must be at least rank 2, but got rank ${e.rank}.`),R(n===3,()=>`Error in RGBToGrayscale: last dimension of an RGB image should be size 3, but got size ${n}.`);const r=e.dtype,i=De(e,"float32"),o=At([.2989,.587,.114]);let a;switch(e.rank){case 2:a=$i("ij,j->i",i,o);break;case 3:a=$i("ijk,k->ij",i,o);break;case 4:a=$i("ijkl,l->ijk",i,o);break;case 5:a=$i("ijklm,m->ijkl",i,o);break;case 6:a=$i("ijklmn,n->ijklm",i,o);break;default:throw new Error("Not a valid tensor rank.")}return a=jn(a,-1),De(a,r)}const r_=J({rgbToGrayscale_:s_});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function i_(s,e,t=0,n=.5){const r=H(s,"image","rotateWithOffset","float32");R(r.rank===4,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${r.rank}.`);const i={image:r},o={radians:e,fillValue:t,center:n};return K.runKernel(Xb,i,o)}const o_=J({rotateWithOffset_:i_});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _i(s,e,t,n,r,i){n==null&&(n=.5),r==null&&(r=Number.NEGATIVE_INFINITY),i==null&&(i=0);const o=s.shape[0];return t=Math.min(t,o),R(0<=n&&n<=1,()=>`iouThreshold must be in [0, 1], but was '${n}'`),R(s.rank===2,()=>`boxes must be a 2D tensor, but was of rank '${s.rank}'`),R(s.shape[1]===4,()=>`boxes must have 4 columns, but 2nd dimension was ${s.shape[1]}`),R(e.rank===1,()=>"scores must be a 1D tensor"),R(e.shape[0]===o,()=>`scores has incompatible shape with boxes. Expected ${o}, but was ${e.shape[0]}`),R(0<=i&&i<=1,()=>`softNmsSigma must be in [0, 1], but was '${i}'`),{maxOutputSize:t,iouThreshold:n,scoreThreshold:r,softNmsSigma:i}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function a_(s,e,t,n=.5,r=Number.NEGATIVE_INFINITY){const i=H(s,"boxes","nonMaxSuppression","float32"),o=H(e,"scores","nonMaxSuppression","float32"),a=_i(i,o,t,n,r);t=a.maxOutputSize,n=a.iouThreshold,r=a.scoreThreshold;const l={maxOutputSize:t,iouThreshold:n,scoreThreshold:r};return K.runKernel(xb,{boxes:i,scores:o},l)}const l_=J({nonMaxSuppression_:a_});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function u_(s,e,t){const n=c_(s,e,t),r=n<0?-(n+1):n;s.splice(r,0,e)}function c_(s,e,t){return f_(s,e,t||h_)}function h_(s,e){return s>e?1:s<e?-1:0}function f_(s,e,t){let n=0,r=s.length,i=0,o=!1;for(;n<r;){i=n+(r-n>>>1);const a=t(e,s[i]);a>0?n=i+1:(r=i,o=!a)}return o?n:-n-1}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function d_(s,e,t,n,r){return Dc(s,e,t,n,r,0)}function p_(s,e,t,n,r,i){return Dc(s,e,t,n,r,0,!1,i,!0)}function m_(s,e,t,n,r,i){return Dc(s,e,t,n,r,i,!0)}function Dc(s,e,t,n,r,i,o=!1,a=!1,l=!1){const u=[];for(let m=0;m<e.length;m++)e[m]>r&&u.push({score:e[m],boxIndex:m,suppressBeginIndex:0});u.sort(af);const c=i>0?-.5/i:0,h=[],d=[];for(;h.length<t&&u.length>0;){const m=u.pop(),{score:S,boxIndex:b,suppressBeginIndex:f}=m;if(S<r)break;let v=!1;for(let _=h.length-1;_>=f;--_){const E=g_(s,b,h[_]);if(E>=n){v=!0;break}if(m.score=m.score*y_(n,c,E),m.score<=r)break}m.suppressBeginIndex=h.length,v||(m.score===S?(h.push(b),d.push(m.score)):m.score>r&&u_(u,m,af))}const w=h.length,k=t-w;a&&k>0&&(h.push(...new Array(k).fill(0)),d.push(...new Array(k).fill(0)));const A={selectedIndices:h};return o&&(A.selectedScores=d),l&&(A.validOutputs=w),A}function g_(s,e,t){const n=s.subarray(e*4,e*4+4),r=s.subarray(t*4,t*4+4),i=Math.min(n[0],n[2]),o=Math.min(n[1],n[3]),a=Math.max(n[0],n[2]),l=Math.max(n[1],n[3]),u=Math.min(r[0],r[2]),c=Math.min(r[1],r[3]),h=Math.max(r[0],r[2]),d=Math.max(r[1],r[3]),w=(a-i)*(l-o),k=(h-u)*(d-c);if(w<=0||k<=0)return 0;const A=Math.max(i,u),m=Math.max(o,c),S=Math.min(a,h),b=Math.min(l,d),f=Math.max(S-A,0)*Math.max(b-m,0);return f/(w+k-f)}function y_(s,e,t){const n=Math.exp(e*t*t);return t<=s?n:0}function af(s,e){return s.score-e.score||s.score===e.score&&e.boxIndex-s.boxIndex}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function b_(s,e,t,n=.5,r=Number.NEGATIVE_INFINITY){const i=H(s,"boxes","nonMaxSuppressionAsync"),o=H(e,"scores","nonMaxSuppressionAsync"),a=_i(i,o,t,n,r);t=a.maxOutputSize,n=a.iouThreshold,r=a.scoreThreshold;const l=await Promise.all([i.data(),o.data()]),u=l[0],c=l[1],{selectedIndices:h}=d_(u,c,t,n,r);return i!==s&&i.dispose(),o!==e&&o.dispose(),At(h,"int32")}const w_=b_;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function x_(s,e,t,n=.5,r=Number.NEGATIVE_INFINITY,i=0){const o=H(s,"boxes","nonMaxSuppression"),a=H(e,"scores","nonMaxSuppression"),l=_i(o,a,t,n,r,i);t=l.maxOutputSize,n=l.iouThreshold,r=l.scoreThreshold,i=l.softNmsSigma;const u={boxes:o,scores:a},c={maxOutputSize:t,iouThreshold:n,scoreThreshold:r,softNmsSigma:i},h=K.runKernel(vb,u,c);return{selectedIndices:h[0],selectedScores:h[1]}}const __=J({nonMaxSuppressionWithScore_:x_});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function v_(s,e,t,n=.5,r=Number.NEGATIVE_INFINITY,i=0){const o=H(s,"boxes","nonMaxSuppressionAsync"),a=H(e,"scores","nonMaxSuppressionAsync"),l=_i(o,a,t,n,r,i);t=l.maxOutputSize,n=l.iouThreshold,r=l.scoreThreshold,i=l.softNmsSigma;const u=await Promise.all([o.data(),a.data()]),c=u[0],h=u[1],{selectedIndices:d,selectedScores:w}=m_(c,h,t,n,r,i);return o!==s&&o.dispose(),a!==e&&a.dispose(),{selectedIndices:At(d,"int32"),selectedScores:At(w)}}const S_=v_;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function k_(s,e,t,n=.5,r=Number.NEGATIVE_INFINITY,i=!1){const o=H(s,"boxes","nonMaxSuppression"),a=H(e,"scores","nonMaxSuppression"),l=_i(o,a,t,n,r,null),u=l.maxOutputSize,c=l.iouThreshold,h=l.scoreThreshold,d={boxes:o,scores:a},w={maxOutputSize:u,iouThreshold:c,scoreThreshold:h,padToMaxOutputSize:i},k=K.runKernel(_b,d,w);return{selectedIndices:k[0],validOutputs:k[1]}}const I_=J({nonMaxSuppressionPadded_:k_});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function E_(s,e,t,n=.5,r=Number.NEGATIVE_INFINITY,i=!1){const o=H(s,"boxes","nonMaxSuppressionAsync"),a=H(e,"scores","nonMaxSuppressionAsync"),l=_i(o,a,t,n,r,null),u=l.maxOutputSize,c=l.iouThreshold,h=l.scoreThreshold,[d,w]=await Promise.all([o.data(),a.data()]),{selectedIndices:k,validOutputs:A}=p_(d,w,u,c,h,i);return o!==s&&o.dispose(),a!==e&&a.dispose(),{selectedIndices:At(k,"int32"),validOutputs:Qt(A,"int32")}}const T_=E_;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function A_(s,e,t=!1,n=!1){const r=H(s,"images","resizeBilinear");R(r.rank===3||r.rank===4,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${r.rank}.`),R(e.length===2,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${e}.`),R(n===!1||t===!1,()=>"Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.");let i=r,o=!1;r.rank===3&&(o=!0,i=ae(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const a={images:i},l={alignCorners:t,halfPixelCenters:n,size:e},u=K.runKernel(Db,a,l);return o?ae(u,[u.shape[1],u.shape[2],u.shape[3]]):u}const C_=J({resizeBilinear_:A_});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function N_(s,e,t=!1,n=!1){const r=H(s,"images","resizeNearestNeighbor");R(r.rank===3||r.rank===4,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${r.rank}.`),R(e.length===2,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${e}.`),R(r.dtype==="float32"||r.dtype==="int32",()=>"`images` must have `int32` or `float32` as dtype"),R(n===!1||t===!1,()=>"Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.");let i=r,o=!1;r.rank===3&&(o=!0,i=ae(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const a={images:i},l={alignCorners:t,halfPixelCenters:n,size:e},u=K.runKernel(rp,a,l);return o?ae(u,[u.shape[1],u.shape[2],u.shape[3]]):u}const $_=J({resizeNearestNeighbor_:N_});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function D_(s,e="binary",t=!1,n=.5){const r=H(s,"image","threshold"),i=.2989,o=.587,a=.114,l=r.shape[0]*r.shape[1];let u=ne(At([n]),255),c,h,d,w;if(R(r.rank===3,()=>`Error in threshold: image must be rank 3,but got rank ${r.rank}.`),R(r.shape[2]===3||r.shape[2]===1,()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${r.shape[2]}.`),R(r.dtype==="int32"||r.dtype==="float32",()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${r.dtype}.`),R(e==="otsu"||e==="binary",()=>`Method must be binary or otsu, but was ${e}`),r.shape[2]===3){[c,h,d]=Gp(r,[1,1,1],-1);const m=ne(c,i),S=ne(h,o),b=ne(d,a);w=he(he(m,S),b)}else w=s;if(e==="otsu"){const m=Nw(De(I2(w),"int32"),ma([]),256);u=O_(m,l)}const k=t?Rp(w,u):$o(w,u);return De(ne(k,255),"int32")}function O_(s,e){let t=At([-1]),n=At([0]),r=At([0]),i,o,a,l,u,c;for(let h=0;h<s.size-1;h++){i=dt(s,0,h+1),o=dt(s,h+1),u=_e(Ae(i),e),c=_e(Ae(o),e);const d=Ae(ne(i,ja(0,i.size)));a=_e(d,Ae(i));const w=Dl(o.shape,i.size),k=he(ja(0,o.size),w),A=ne(o,k);l=_e(Ae(A),Ae(o));const m=ke(a,l),S=ke(a,l),b=ne(u,c);r=ne(ne(b,m),S);const f=$o(r,n);n=mr(f,r,n),t=mr(f,At([h]),t)}return t}const M_=J({threshold_:D_});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function P_(s,e,t="nearest",n="constant",r=0,i){const o=H(s,"image","transform","float32"),a=H(e,"transforms","transform","float32");R(o.rank===4,()=>`Error in transform: image must be rank 4,but got rank ${o.rank}.`),R(a.rank===2&&(a.shape[0]===o.shape[0]||a.shape[0]===1)&&a.shape[1]===8,()=>"Error in transform: Input transform should be batch x 8 or 1 x 8"),R(i==null||i.length===2,()=>`Error in transform: outputShape must be [height, width] or null, but got ${i}.`);const l={image:o,transforms:a},u={interpolation:t,fillMode:n,fillValue:r,outputShape:i};return K.runKernel(qb,l,u)}const R_=J({transform_:P_});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function L_(s,e,t){const n=H(s,"a","bandPart");R(n.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${n.rank}.`);const r=n.shape,[i,o]=n.shape.slice(-2);let a,l;typeof e=="number"?(R(e%1===0,()=>`bandPart(): numLower must be an integer, got ${e}.`),R(e<=i,()=>`bandPart(): numLower (${e}) must not be greater than the number of rows (${i}).`),a=H(e<0?i:e,"numLower","bandPart")):(R(e.dtype==="int32",()=>"bandPart(): numLower's dtype must be an int32."),a=mr(Zh(e,0),i,Ha(e,i))),typeof t=="number"?(R(t%1===0,()=>`bandPart(): numUpper must be an integer, got ${t}.`),R(t<=o,()=>`bandPart(): numUpper (${t}) must not be greater than the number of columns (${o}).`),l=H(t<0?o:t,"numUpper","bandPart")):(R(t.dtype==="int32",()=>"bandPart(): numUpper's dtype must be an int32."),l=mr(Zh(t,0),o,Ha(t,o)));const u=ae(ja(0,i,1,"int32"),[-1,1]),c=ja(0,o,1,"int32"),h=ke(u,c),d=Ol(Rp(h,a),vx(h,wi(l))),w=fi([i,o],n.dtype);return ae(Ka(qp(ae(n,[-1,i,o])).map(k=>mr(d,k,w))),r)}const B_=J({bandPart_:L_});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function F_(s){let e;if(Array.isArray(s)){e=!1,R(s!=null&&s.length>0,()=>"Gram-Schmidt process: input must not be null, undefined, or empty");const r=s[0].shape[0];for(let i=1;i<s.length;++i)R(s[i].shape[0]===r,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${s[i].shape[0]} vs. ${r})`)}else e=!0,s=Gp(s,s.shape[0],0).map(r=>Ml(r,[0]));R(s.length<=s[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${s.length}) exceeds number of dimensions (${s[0].shape[0]}).`);const t=[],n=s;for(let r=0;r<s.length;++r)t.push(K.tidy(()=>{let i=n[r];if(r>0)for(let o=0;o<r;++o){const a=ne(Ae(ne(t[o],i)),t[o]);i=ke(i,a)}return _e(i,Mp(i,"euclidean"))}));return e?Ka(t,0):t}const U_=J({gramSchmidt_:F_});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function z_(s,e=!1){if(R(s.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${s.rank}`),s.rank===2)return lf(s,e);{const t=s.shape.slice(0,s.shape.length-2).reduce((l,u)=>l*u),n=qp(ae(s,[t,s.shape[s.shape.length-2],s.shape[s.shape.length-1]]),0),r=[],i=[];n.forEach(l=>{const[u,c]=lf(l,e);r.push(u),i.push(c)});const o=ae(Ka(r,0),s.shape),a=ae(Ka(i,0),s.shape);return[o,a]}}function lf(s,e=!1){return K.tidy(()=>{R(s.shape.length===2,()=>`qr2d() requires a 2D Tensor, but got a ${s.shape.length}D Tensor.`);const t=s.shape[0],n=s.shape[1];let r=Pp(t),i=dr(s);const o=su([[1]],[1,1]);let a=dr(o);const l=t>=n?n:t;for(let u=0;u<l;++u){const c=i,h=a,d=r;[a,i,r]=K.tidy(()=>{const w=dt(i,[u,u],[t-u,1]),k=Mp(w),A=dt(i,[u,u],[1,1]),m=mr($o(A,0),su([[-1]]),su([[1]])),S=ke(A,ne(m,k)),b=_e(w,S);b.shape[0]===1?a=dr(o):a=pr([o,dt(b,[1,0],[b.shape[0]-1,b.shape[1]])],0);const f=wi(_e(Pn(m,S),k)),v=dt(i,[u,0],[t-u,n]),_=ne(f,a),E=He(a);if(u===0)i=ke(v,Pn(_,Pn(E,v)));else{const $=ke(v,Pn(_,Pn(E,v)));i=pr([dt(i,[0,0],[u,n]),$],0)}const D=He(_),M=dt(r,[0,u],[t,r.shape[1]-u]);if(u===0)r=ke(M,Pn(Pn(M,a),D));else{const $=ke(M,Pn(Pn(M,a),D));r=pr([dt(r,[0,0],[t,u]),$],1)}return[a,i,r]}),Pe([c,h,d])}return!e&&t>n&&(r=dt(r,[0,0],[t,n]),i=dt(i,[0,0],[n,n])),[r,i]})}const W_=J({qr_:z_});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Yo={flipLeftRight:e_,grayscaleToRGB:n_,resizeNearestNeighbor:$_,resizeBilinear:C_,rgbToGrayscale:r_,rotateWithOffset:o_,cropAndResize:Q2,nonMaxSuppression:l_,nonMaxSuppressionAsync:w_,nonMaxSuppressionWithScore:__,nonMaxSuppressionWithScoreAsync:S_,nonMaxSuppressionPadded:I_,nonMaxSuppressionPaddedAsync:T_,threshold:M_,transform:R_},G_={bandPart:B_,gramSchmidt:U_,qr:W_};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const V_=new Map,q_=new Map;class vi{getClassName(){return this.constructor.className}static fromConfig(e,t){return new e(t)}}class on{constructor(){this.classNameMap={}}static getMap(){return on.instance==null&&(on.instance=new on),on.instance}static register(e){on.getMap().classNameMap[e.className]=[e,e.fromConfig]}}function le(s,e,t){R(s.className!=null,()=>"Class being registered does not have the static className property defined."),R(typeof s.className=="string",()=>"className is required to be a string, but got type "+typeof s.className),R(s.className.length>0,()=>"Class being registered has an empty-string as its className, which is disallowed."),typeof e>"u"&&(e="Custom"),typeof t>"u"&&(t=s.className);const n=t,r=e+">"+n;return on.register(s),V_.set(r,s),q_.set(s,r),s}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Bs extends vi{minimize(e,t=!1,n){const{value:r,grads:i}=this.computeGradients(e,n);if(n!=null){const o=n.map(a=>({name:a.name,tensor:i[a.name]}));this.applyGradients(o)}else this.applyGradients(i);return Pe(i),t?r:(r.dispose(),null)}get iterations(){return this.iterations_==null&&(this.iterations_=0),this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(e,t){return Dx(e,t)}dispose(){this.iterations_!=null&&Pe(this.iterations_)}async saveIterations(){return this.iterations_==null&&(this.iterations_=0),{name:"iter",tensor:Qt(this.iterations_,"int32")}}async getWeights(){throw new Error("getWeights() is not implemented for this optimizer yet.")}async setWeights(e){throw new Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(e){return this.iterations_=(await e[0].tensor.data())[0],e.slice(1)}}Object.defineProperty(Bs,Symbol.hasInstance,{value:s=>s.minimize!=null&&s.computeGradients!=null&&s.applyGradients!=null});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Hp extends Bs{static get className(){return"Adadelta"}constructor(e,t,n=null){super(),this.learningRate=e,this.rho=t,this.epsilon=n,this.accumulatedGrads=[],this.accumulatedUpdates=[],n==null&&(this.epsilon=K.backend.epsilon())}applyGradients(e){(Array.isArray(e)?e.map(n=>n.name):Object.keys(e)).forEach((n,r)=>{const i=K.registeredVariables[n],o=!1;this.accumulatedGrads[r]==null&&(this.accumulatedGrads[r]={originalName:`${n}/accum_grad`,variable:Q(()=>Qn(i).variable(o))}),this.accumulatedUpdates[r]==null&&(this.accumulatedUpdates[r]={originalName:`${n}/accum_var`,variable:Q(()=>Qn(i).variable(o))});const a=Array.isArray(e)?e[r].tensor:e[n];if(a==null)return;const l=this.accumulatedGrads[r].variable,u=this.accumulatedUpdates[r].variable;Q(()=>{const c=he(ne(l,this.rho),ne(Os(a),1-this.rho)),h=ne(_e(Cn(he(u,this.epsilon)),Cn(he(l,this.epsilon))),a),d=he(ne(u,this.rho),ne(Os(h),1-this.rho));l.assign(c),u.assign(d);const w=he(ne(h,-this.learningRate),i);i.assign(w)})}),this.incrementIterations()}dispose(){this.accumulatedUpdates!=null&&(Pe(this.accumulatedGrads.map(e=>e.variable)),Pe(this.accumulatedUpdates.map(e=>e.variable)))}async getWeights(){const e=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(e.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(e){e=await this.extractIterations(e);const t=e.length/2,n=!1;this.accumulatedGrads=e.slice(0,t).map(r=>({originalName:r.name,variable:r.tensor.variable(n)})),this.accumulatedUpdates=e.slice(t,t*2).map(r=>({originalName:r.name,variable:r.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.rho,t.epsilon)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class jp extends Bs{static get className(){return"Adagrad"}constructor(e,t=.1){super(),this.learningRate=e,this.initialAccumulatorValue=t,this.accumulatedGrads=[]}applyGradients(e){(Array.isArray(e)?e.map(n=>n.name):Object.keys(e)).forEach((n,r)=>{const i=K.registeredVariables[n];this.accumulatedGrads[r]==null&&(this.accumulatedGrads[r]={originalName:`${n}/accumulator`,variable:Q(()=>Dl(i.shape,this.initialAccumulatorValue).variable(!1))});const o=Array.isArray(e)?e[r].tensor:e[n];if(o==null)return;const a=this.accumulatedGrads[r].variable;Q(()=>{const l=he(a,Os(o));a.assign(l);const u=he(ne(_e(o,Cn(he(l,K.backend.epsilon()))),-this.learningRate),i);i.assign(u)})}),this.incrementIterations()}dispose(){this.accumulatedGrads!=null&&Pe(this.accumulatedGrads.map(e=>e.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);const t=!1;this.accumulatedGrads=e.map(n=>({originalName:n.name,variable:n.tensor.variable(t)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(e,t){return new e(t.learningRate,t.initialAccumulatorValue)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Kp extends Bs{static get className(){return"Adam"}constructor(e,t,n,r=null){super(),this.learningRate=e,this.beta1=t,this.beta2=n,this.epsilon=r,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],Q(()=>{this.accBeta1=Qt(t).variable(),this.accBeta2=Qt(n).variable()}),r==null&&(this.epsilon=K.backend.epsilon())}applyGradients(e){const t=Array.isArray(e)?e.map(n=>n.name):Object.keys(e);Q(()=>{const n=ke(1,this.accBeta1),r=ke(1,this.accBeta2);t.forEach((i,o)=>{const a=K.registeredVariables[i],l=!1;this.accumulatedFirstMoment[o]==null&&(this.accumulatedFirstMoment[o]={originalName:`${i}/m`,variable:Q(()=>Qn(a).variable(l))}),this.accumulatedSecondMoment[o]==null&&(this.accumulatedSecondMoment[o]={originalName:`${i}/v`,variable:Q(()=>Qn(a).variable(l))});const u=Array.isArray(e)?e[o].tensor:e[i];if(u==null)return;const c=this.accumulatedFirstMoment[o].variable,h=this.accumulatedSecondMoment[o].variable,d=he(ne(c,this.beta1),ne(u,1-this.beta1)),w=he(ne(h,this.beta2),ne(Os(u),1-this.beta2)),k=_e(d,n),A=_e(w,r);c.assign(d),h.assign(w);const m=he(ne(_e(k,he(Cn(A),this.epsilon)),-this.learningRate),a);a.assign(m)}),this.accBeta1.assign(ne(this.accBeta1,this.beta1)),this.accBeta2.assign(ne(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),this.accumulatedFirstMoment!=null&&Pe(this.accumulatedFirstMoment.map(e=>e.variable)),this.accumulatedSecondMoment!=null&&Pe(this.accumulatedSecondMoment.map(e=>e.variable))}async getWeights(){const e=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(e.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(e){e=await this.extractIterations(e),Q(()=>{this.accBeta1.assign(qa(this.beta1,this.iterations_+1)),this.accBeta2.assign(qa(this.beta2,this.iterations_+1))});const t=e.length/2,n=!1;this.accumulatedFirstMoment=e.slice(0,t).map(r=>({originalName:r.name,variable:r.tensor.variable(n)})),this.accumulatedSecondMoment=e.slice(t,t*2).map(r=>({originalName:r.name,variable:r.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Xp extends Bs{static get className(){return"Adamax"}constructor(e,t,n,r=null,i=0){super(),this.learningRate=e,this.beta1=t,this.beta2=n,this.epsilon=r,this.decay=i,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],Q(()=>{this.iteration=Qt(0).variable(),this.accBeta1=Qt(t).variable()}),r==null&&(this.epsilon=K.backend.epsilon())}applyGradients(e){const t=Array.isArray(e)?e.map(n=>n.name):Object.keys(e);Q(()=>{const n=ke(1,this.accBeta1),r=_e(-this.learningRate,he(ne(this.iteration,this.decay),1));t.forEach((i,o)=>{const a=K.registeredVariables[i],l=!1;this.accumulatedFirstMoment[o]==null&&(this.accumulatedFirstMoment[o]={originalName:`${i}/m`,variable:Qn(a).variable(l)}),this.accumulatedWeightedInfNorm[o]==null&&(this.accumulatedWeightedInfNorm[o]={originalName:`${i}/v`,variable:Qn(a).variable(l)});const u=Array.isArray(e)?e[o].tensor:e[i];if(u==null)return;const c=this.accumulatedFirstMoment[o].variable,h=this.accumulatedWeightedInfNorm[o].variable,d=he(ne(c,this.beta1),ne(u,1-this.beta1)),w=ne(h,this.beta2),k=Tt(u),A=xi(w,k);c.assign(d),h.assign(A);const m=he(ne(_e(r,n),_e(d,he(A,this.epsilon))),a);a.assign(m)}),this.iteration.assign(he(this.iteration,1)),this.accBeta1.assign(ne(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),this.accumulatedFirstMoment!=null&&Pe(this.accumulatedFirstMoment.map(e=>e.variable)),this.accumulatedWeightedInfNorm!=null&&Pe(this.accumulatedWeightedInfNorm.map(e=>e.variable))}async getWeights(){throw new Error("getWeights() is not implemented for Adamax yet.")}async setWeights(e){throw new Error("setWeights() is not implemented for Adamax yet.")}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon,t.decay)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Oc extends Bs{static get className(){return"SGD"}constructor(e){super(),this.learningRate=e,this.setLearningRate(e)}applyGradients(e){(Array.isArray(e)?e.map(n=>n.name):Object.keys(e)).forEach((n,r)=>{const i=Array.isArray(e)?e[r].tensor:e[n];if(i==null)return;const o=K.registeredVariables[n];Q(()=>{const a=he(ne(this.c,i),o);o.assign(a)})}),this.incrementIterations()}setLearningRate(e){this.learningRate=e,this.c!=null&&this.c.dispose(),this.c=oi(Qt(-e))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(e){if(e=await this.extractIterations(e),e.length!==0)throw new Error("SGD optimizer does not have settable weights.")}getConfig(){return{learningRate:this.learningRate}}static fromConfig(e,t){return new e(t.learningRate)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Yp extends Oc{static get className(){return"Momentum"}constructor(e,t,n=!1){super(e),this.learningRate=e,this.momentum=t,this.useNesterov=n,this.accumulations=[],this.m=Qt(this.momentum)}applyGradients(e){(Array.isArray(e)?e.map(n=>n.name):Object.keys(e)).forEach((n,r)=>{const i=K.registeredVariables[n];this.accumulations[r]==null&&(this.accumulations[r]={originalName:`${n}/momentum`,variable:Q(()=>Qn(i).variable(!1))});const o=this.accumulations[r].variable,a=Array.isArray(e)?e[r].tensor:e[n];a!=null&&Q(()=>{let l;const u=he(ne(this.m,o),a);this.useNesterov?l=he(ne(this.c,he(a,ne(u,this.m))),i):l=he(ne(this.c,u),i),o.assign(u),i.assign(l)})}),this.incrementIterations()}dispose(){this.m.dispose(),this.accumulations!=null&&Pe(this.accumulations.map(e=>e.variable))}setMomentum(e){this.momentum=e}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);const t=!1;this.accumulations=e.map(n=>({originalName:n.name,variable:n.tensor.variable(t)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(e,t){return new e(t.learningRate,t.momentum,t.useNesterov)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Zp extends Bs{static get className(){return"RMSProp"}constructor(e,t=.9,n=0,r=null,i=!1){if(super(),this.learningRate=e,this.decay=t,this.momentum=n,this.epsilon=r,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=i,r==null&&(this.epsilon=K.backend.epsilon()),e==null)throw new Error("learningRate for RMSPropOptimizer must be defined.")}applyGradients(e){(Array.isArray(e)?e.map(n=>n.name):Object.keys(e)).forEach((n,r)=>{const i=K.registeredVariables[n],o=!1;this.accumulatedMeanSquares[r]==null&&(this.accumulatedMeanSquares[r]={originalName:`${n}/rms`,variable:Q(()=>Qn(i).variable(o))}),this.accumulatedMoments[r]==null&&(this.accumulatedMoments[r]={originalName:`${n}/momentum`,variable:Q(()=>Qn(i).variable(o))}),this.accumulatedMeanGrads[r]==null&&this.centered&&(this.accumulatedMeanGrads[r]={originalName:`${n}/mg`,variable:Q(()=>Qn(i).variable(o))});const a=Array.isArray(e)?e[r].tensor:e[n];if(a==null)return;const l=this.accumulatedMeanSquares[r].variable,u=this.accumulatedMoments[r].variable;Q(()=>{const c=he(ne(l,this.decay),ne(Os(a),1-this.decay));if(this.centered){const h=this.accumulatedMeanGrads[r].variable,d=he(ne(h,this.decay),ne(a,1-this.decay)),w=_e(ne(a,this.learningRate),Cn(ke(c,he(Os(d),this.epsilon)))),k=he(ne(u,this.momentum),w);l.assign(c),h.assign(d),u.assign(k);const A=ke(i,k);i.assign(A)}else{const h=he(ne(l,this.decay),ne(Os(a),1-this.decay)),d=he(ne(u,this.momentum),_e(ne(a,this.learningRate),Cn(he(h,this.epsilon))));l.assign(h),u.assign(d);const w=ke(i,d);i.assign(w)}})}),this.incrementIterations()}dispose(){this.accumulatedMeanSquares!=null&&Pe(this.accumulatedMeanSquares.map(e=>e.variable)),this.accumulatedMeanGrads!=null&&this.centered&&Pe(this.accumulatedMeanGrads.map(e=>e.variable)),this.accumulatedMoments!=null&&Pe(this.accumulatedMoments.map(e=>e.variable))}async getWeights(){const e=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&e.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(e.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(e){e=await this.extractIterations(e);const t=this.centered?e.length/3:e.length/2,n=!1;this.accumulatedMeanSquares=e.slice(0,t).map(r=>({originalName:r.name,variable:r.tensor.variable(n)})),this.accumulatedMoments=e.slice(t,t*2).map(r=>({originalName:r.name,variable:r.tensor.variable(n)})),this.centered&&(this.accumulatedMeanGrads=e.slice(t*2,t*3).map(r=>({originalName:r.name,variable:r.tensor.variable(n)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(e,t){return new e(t.learningRate,t.decay,t.momentum,t.epsilon,t.centered)}}/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const H_=[Hp,jp,Kp,Xp,Yp,Zp,Oc];function j_(){for(const s of H_)le(s)}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function K_(s,e,t){const n=s.shape.length;R(n===e.length,()=>`Error in slice${n}D: Length of begin ${e} must match the rank of the array (${n}).`),R(n===t.length,()=>`Error in slice${n}D: Length of size ${t} must match the rank of the array (${n}).`);for(let r=0;r<n;++r)R(e[r]+t[r]<=s.shape[r],()=>`Error in slice${n}D: begin[${r}] + size[${r}] (${e[r]+t[r]}) would overflow input.shape[${r}] (${s.shape[r]})`)}function X_(s,e,t){let n=t.length;for(let r=0;r<t.length;r++)if(t[r]>1){n=r;break}for(let r=n+1;r<t.length;r++)if(e[r]>0||t[r]!==s[r])return!1;return!0}function Y_(s,e){let t=s.length>0?s[s.length-1]:1;for(let n=0;n<s.length-1;n++)t+=s[n]*e[n];return t}function Z_(s,e,t){let n;const r=s.shape.length;typeof e=="number"?n=[e,...new Array(r-1).fill(0)]:e.length<r?n=e.concat(new Array(r-e.length).fill(0)):n=e.slice(),n.forEach(o=>{R(o!==-1,()=>"slice() does not support negative begin indexing.")});let i;return t==null?i=new Array(r).fill(-1):typeof t=="number"?i=[t,...new Array(r-1).fill(-1)]:t.length<r?i=t.concat(new Array(r-t.length).fill(-1)):i=t,i=i.map((o,a)=>o>=0?o:(R(o===-1,()=>`Negative size values should be exactly -1 but got ${o} for the slice() size at index ${a}.`),s.shape[a]-n[a])),[n,i]}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Q_{static sgd(e){return new Oc(e)}static momentum(e,t,n=!1){return new Yp(e,t,n)}static rmsprop(e,t=.9,n=0,r=null,i=!1){return new Zp(e,t,n,r,i)}static adam(e=.001,t=.9,n=.999,r=null){return new Kp(e,t,n,r)}static adadelta(e=.001,t=.95,n=null){return new Hp(e,t,n)}static adamax(e=.002,t=.9,n=.999,r=null,i=0){return new Xp(e,t,n,r,i)}static adagrad(e,t=.1){return new jp(e,t)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $r=Q_;/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const J_=typeof requestAnimationFrame<"u"?requestAnimationFrame:typeof setImmediate<"u"?setImmediate:s=>s();function ev(){return new Promise(s=>J_(()=>s()))}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tv(s,e){const t=s[0].length;s.forEach((r,i)=>{R(r.length===t,()=>`Error in concat${t}D: rank of tensors[${i}] must be the same as the rank of the rest (${t})`)}),R(e>=0&&e<t,()=>`Error in concat${t}D: axis must be between 0 and ${t-1}.`);const n=s[0];s.forEach((r,i)=>{for(let o=0;o<t;o++)R(o===e||r[o]===n[o],()=>`Error in concat${t}D: Shape of tensors[${i}] (${r}) does not match the shape of the rest (${n}) along the non-concatenated axis ${i}.`)})}function xo(s,e){const t=s[0].slice();for(let n=1;n<s.length;n++)t[e]+=s[n][e];return t}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Kn;(function(s){s[s.FIRST_DIM_SIZE=0]="FIRST_DIM_SIZE",s[s.VALUE_ROWIDS=1]="VALUE_ROWIDS",s[s.ROW_LENGTHS=2]="ROW_LENGTHS",s[s.ROW_SPLITS=3]="ROW_SPLITS",s[s.ROW_LIMITS=4]="ROW_LIMITS",s[s.ROW_STARTS=5]="ROW_STARTS"})(Kn||(Kn={}));function nv(s,e,t){let n=new Array;if(t==null&&e==null)return n;if(e==null)for(;n.length<s+t.length;)n.push(-1);else n=e.slice();if(t==null)return n;if(s+t.length!==n.length)throw new Error(`rt input.shape and shape=${e} are incompatible: rt input.rank = ${s+t.length}, but shape.rank = ${n.length}`);for(let r=1;r<t.length;++r){const i=t[r],o=n[n.length-t.length+r],a=n[o];if(i>=0)if(a>=0){if(a!==i)throw new Error(`rt input.shape and shape=${e} are incompatible: rt input.shape[${r+s}] = ${i} but shape[${r+s}] = ${a}`)}else n[o]=i}return n}function sv(s){const e={FIRST_DIM_SIZE:Kn.FIRST_DIM_SIZE,VALUE_ROWIDS:Kn.VALUE_ROWIDS,ROW_LENGTHS:Kn.ROW_LENGTHS,ROW_SPLITS:Kn.ROW_SPLITS,ROW_LIMITS:Kn.ROW_LIMITS,ROW_STARTS:Kn.ROW_STARTS},t=[];for(const n of s)if(n in e)t.push(e[n]);else break;return t}function rv(s){return s.length===0?0:s[0]===Kn.FIRST_DIM_SIZE?s.length-1:s.length}function iv(s,e){if(s==null||e==null)return;const t=s.length,n=e.length;if(t>=n)throw new Error(`defaultValue.shape=${s} and ragged tensor flatValues.shape=${e}, are incompatible: defaultValue.rank = ${t} must be less than ragged tensor input flatValues.rank = ${n})`);for(let r=0;r<Math.min(t,n-1);++r){const i=s[r],o=e[r+1];if(i>=0&&o>=0&&i!==1&&i!==o)throw new Error(`defaultValue.shape=${s}, and ragged tensor input flatValues.shape=${e} are incompatible: defaultValue.shape[${r-s.length}] = ${i} but ragged tensor input.flatValues.shape[${r-s.length}] = ${o}`)}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ov=1.7580993408473768,av=1.0507009873554805;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lv=.3275911,uv=.254829592,cv=-.284496736,hv=1.421413741,fv=-1.453152027,dv=1.061405429;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uf(s,e){if(s.length!==e.length)throw new Error(`Cannot merge real and imag arrays of different lengths. real:${s.length}, imag: ${e.length}.`);const t=new Float32Array(s.length*2);for(let n=0;n<t.length;n+=2)t[n]=s[n/2],t[n+1]=e[n/2];return t}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pv(s){return`Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${s}`}function mv(s,e){return`indices(${s}, 0) is invalid: ${e} < 0`}function gv(s,e,t){return`indices(${s}, 0) is invalid: ${e} >= ${t}`}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yv(s,e){return`only one output dimension may be -1, not both ${s} and ${e}`}function bv(s,e){return`size ${s} must be non-negative, not ${e}`}function wv(){return"reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero"}function xv(s,e){const t=me(s),n=me(e);return`Input to reshape is a SparseTensor with ${t}
  dense values, but the requested shape requires a multiple of ${n}. inputShape=${s} outputShape= ${e}`}function _v(s,e){const t=me(s),n=me(e);return`Input to reshape is a tensor with ${t} dense values, but the requested shape has ${n}. inputShape=${s} outputShape=${e}`}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cf(){return"segment ids must be >= 0"}function vv(){return"segment ids are not increasing"}function Sv(s,e){return`Segment id ${s} out of range [0, ${e}), possibly because segmentIds input is not sorted.`}function kv(s,e,t){return`Bad: indices[${s}] == ${e} out of range [0, ${t})`}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qp(s){try{return s.map(e=>Fa(e))}catch(e){throw new Error(`Failed to decode encoded string bytes into utf-8, error: ${e}`)}}function Iv(s){return s.map(e=>fr(e))}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */j_();/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */const Ev=["channelsFirst","channelsLast"],Tv=["nearest","bilinear"],Av=["valid","same","causal"],Cv=["max","avg"];/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */class ks extends Error{constructor(e){super(e),Object.setPrototypeOf(this,ks.prototype)}}class Ms extends Error{constructor(e){super(e),Object.setPrototypeOf(this,Ms.prototype)}}class q extends Error{constructor(e){super(e),Object.setPrototypeOf(this,q.prototype)}}class Se extends Error{constructor(e){super(e),Object.setPrototypeOf(this,Se.prototype)}}class Mc extends Error{constructor(e){super(e),Object.setPrototypeOf(this,Mc.prototype)}}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function Xa(s,e){if(Array.isArray(s)){let t=[];for(let n=0;n<e;n++)t=t.concat(s);return t}else{const t=new Array(e);return t.fill(s),t}}function Xn(s,e){if(!s)throw new Mc(e)}function hf(s,e){let t=0;for(const n of s)n===e&&t++;return t}function Rt(s){return s.length===1?s[0]:s}function Ne(s){return Array.isArray(s)?s:[s]}function us(s){const t=s.replace(/(.)([A-Z][a-z0-9]+)/g,"$1_$2").replace(/([a-z])([A-Z])/g,"$1_$2").toLowerCase();return t[0]!=="_"?t:"private"+t}function nr(s){return s.length<=1||s.indexOf("_")===-1?s:s.replace(/[_]+(\w|$)/g,(e,t)=>t.toUpperCase())}let en={};function Pc(s){if(s==null)return null;const e={};return e.className=s.getClassName(),e.config=s.getConfig(),e}function zu(s){if(!(s==null||typeof s!="object"))if(Array.isArray(s))s.forEach(e=>zu(e));else{const e=Object.keys(s);for(const t of e){const n=s[t];n!=null&&typeof n=="object"&&(!Array.isArray(n)&&n.type==="ndarray"&&typeof n.value=="number"?s[t]=n.value:zu(n))}}}function Oo(s,e={},t={},n="object",r=!1){if(typeof s=="string"){const i=s;let o;if(i in t)o=t[i];else if(i in en)o=en[i];else if(o=e[i],o==null)throw new q(`Unknown ${n}: ${s}. This may be due to one of the following reasons:
1. The ${n} is defined in Python, in which case it needs to be ported to TensorFlow.js or your JavaScript code.
2. The custom ${n} is defined in JavaScript, but is not registered properly with tf.serialization.registerClass().`);return o}else{const i=s;if(i.className==null||i.config==null)throw new q(`${n}: Improper config format: ${JSON.stringify(i)}.
'className' and 'config' must set.`);const o=i.className;let a,l;if(o in t?[a,l]=t[o]:o in en?[a,l]=en.className:o in e&&([a,l]=e[o]),a==null)throw new q(`Unknown ${n}: ${o}. This may be due to one of the following reasons:
1. The ${n} is defined in Python, in which case it needs to be ported to TensorFlow.js or your JavaScript code.
2. The custom ${n} is defined in JavaScript, but is not registered properly with tf.serialization.registerClass().`);if(l!=null){const u={};for(const w of Object.keys(en))u[w]=en[w];for(const w of Object.keys(t))u[w]=t[w];const c=i.config;c.customObjects=u;const h=Object.assign({},en);for(const w of Object.keys(t))en[w]=t[w];zu(i.config);const d=l(a,i.config,t,r);return en=Object.assign({},h),d}else{const u=Object.assign({},en);for(const h of Object.keys(t))en[h]=t[h];const c=new a(i.config);return en=Object.assign({},u),c}}}function Nv(s,e){return s<e?-1:s>e?1:0}function Zo(s,e){return-1*Nv(s,e)}function gr(s){if(s==null)return s;const e=[];for(const t of s)e.indexOf(t)===-1&&e.push(t);return e}function $v(s){if(s==null)throw new q(`Invalid value in obj: ${JSON.stringify(s)}`);for(const e in s)if(s.hasOwnProperty(e))return!1;return!0}function Si(s,e,t){if(t!=null&&s.indexOf(t)<0)throw new q(`${t} is not a valid ${e}.  Valid values are ${s} or null/undefined.`)}function Rc(s,e,t=0,n=1/0){return Xn(t>=0),Xn(n>=t),Array.isArray(s)&&s.length>=t&&s.length<=n&&s.every(r=>typeof r===e)}function ys(s,e){Array.isArray(s)?(R(s.length>0,()=>`${e} is unexpectedly an empty array.`),s.forEach((t,n)=>ys(t,`element ${n+1} of ${e}`))):R(Number.isInteger(s)&&s>0,()=>`Expected ${e} to be a positive integer, but got ${Jp(s)}.`)}function Jp(s){return s===null?"null":Array.isArray(s)?"["+s.map(e=>Jp(e)).join(",")+"]":typeof s=="string"?`"${s}"`:`${s}`}function Dv(s,e,t){let n=t!=null?t():ui(),r;return(...o)=>{const a=t!=null?t():ui();return a-n<e||(n=a,r=s(...o)),r}}function Ov(s){return s==="relu"?"relu":s==="linear"?"linear":s==="elu"?"elu":null}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */const Dr=new Map;function Ze(s){Si(Ev,"DataFormat",s)}function Mv(s){Si(Tv,"InterpolationFormat",s)}function hn(s){Si(Av,"PaddingMode",s)}function em(s){Si(Cv,"PoolMode",s)}const to=[],ff="/";function Ia(s,e){to.push(s);try{const t=e();return to.pop(),t}catch(t){throw to.pop(),t}}function Pv(){return to.length===0?"":to.join(ff)+ff}function tm(s){if(!sm(s))throw new Error("Not a valid tensor name: '"+s+"'");return Pv()+s}function nm(s){if(!sm(s))throw new Error("Not a valid tensor name: '"+s+"'");Dr.has(s)||Dr.set(s,0);const e=Dr.get(s);if(Dr.set(s,Dr.get(s)+1),e>0){const t=`${s}_${e}`;return Dr.set(t,1),t}else return s}const Rv=new RegExp(/^[A-Za-z0-9][-A-Za-z0-9\._\/]*$/);function sm(s){return!!s.match(Rv)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function Lv(s){return s===parseInt(s.toString(),10)}function no(s,e,t){e==null&&(e=0),t==null&&(t=s.length);let n=1;for(let r=e;r<t;++r)n*=s[r];return n}function rm(s){if(s.length===0)return Number.NaN;let e=Number.NEGATIVE_INFINITY;for(let t=0;t<s.length;t++){const n=s[t];n>e&&(e=n)}return e}function Ya(s,e){if(e<s)throw new q(`end (${e}) < begin (${s}) is forbidden.`);const t=[];for(let n=s;n<e;++n)t.push(n);return t}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */let ru;function Ye(){return ru==null&&(ru=B1().epsilon()),ru}function ki(){return"channelsLast"}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function im(s,e){return De(s,e)}function Lc(s,e=-1){const t=s.shape.slice();return e<0&&(e=t.length+e+1),t.splice(e,0,1),ae(s,t)}function Bv(s){const e=[no(s.shape)];return ae(s,e)}function yr(s,e,t){return Q(()=>{switch(s.rank){case 1:return Nc(s,e,t);case 2:return zp(s,[e,0],[t,s.shape[1]]);case 3:return $c(s,[e,0,0],[t,s.shape[1],s.shape[2]]);case 4:return bo(s,[e,0,0,0],[t,s.shape[1],s.shape[2],s.shape[3]]);case 5:return dt(s,[e,0,0,0,0],[t,s.shape[1],s.shape[2],s.shape[3],s.shape[4]]);case 6:return dt(s,[e,0,0,0,0,0],[t,s.shape[1],s.shape[2],s.shape[3],s.shape[4],s.shape[5]]);default:throw new q(`sliceAlongFirstAxis() received an unsupported tensor rank: ${s.rank}`)}})}function iu(s,e,t){return Q(()=>{switch(s.rank){case 1:return Nc(s,e,t);case 2:return zp(s,[0,e],[s.shape[0],t]);case 3:return $c(s,[0,0,e],[s.shape[0],s.shape[1],t]);case 4:return bo(s,[0,0,0,e],[s.shape[0],s.shape[1],s.shape[2],t]);default:throw new q(`sliceAlongLastAxis() received an unsupported tensor rank: ${s.rank}`)}})}function Qo(s,e,t,n){return Q(()=>{switch(s.rank){case 1:return Nc(s,e,t);case 2:switch(n){case 1:return yr(s,e,t);case 2:return iu(s,e,t);default:throw new q(`The axis is not within the rank of the tensor ${n}`)}case 3:switch(n){case 1:return yr(s,e,t);case 2:return $c(s,[0,e,0],[s.shape[0],t,s.shape[2]]);case 3:return iu(s,e,t);default:throw new q(`The axis is not within the rank of the tensor ${n}`)}case 4:switch(n){case 1:return yr(s,e,t);case 2:return bo(s,[0,e,0,0],[s.shape[0],t,s.shape[2],s.shape[3]]);case 3:return bo(s,[0,0,e,0],[s.shape[0],s.shape[1],t,s.shape[3]]);case 4:return iu(s,e,t);default:throw new q(`The axis is not within the rank of the tensor ${n}`)}default:throw new q(`sliceAlongLastAxis() received an unsupported tensor rank: ${s.rank}`)}})}function Fv(s,e=-1){let t;return e<0&&(t=s[0].rank,t!==0?e=t:e=0),e===s[0].rank&&(e=-1),pr(s,e)}function om(s,e=0,t=1,n,r){return y2(s,e,t,n,r)}function Uv(s,e,t){return Q(()=>(Array.isArray(e)?e=At(e,"int32"):e=De(e,"int32"),wx(s,e,t)))}function Mo(s){return ne(s,s)}function zv(s,e,t){const n=e.shape;if(e.rank!==1&&e.rank!==s)throw new q(`Unexpected bias dimensions: ${e.rank}; expected it to be 1 or ${s}`);if(s===5){if(t==="channelsFirst")return n.length===1?ae(e,[1,n[0],1,1,1]):ae(e,[1,n[3],n[0],n[1],n[2]]);if(t==="channelsLast")return n.length===1?ae(e,[1,1,1,1,n[0]]):ae(e,[1].concat(n))}else if(s===4){if(t==="channelsFirst")return n.length===1?ae(e,[1,n[0],1,1]):ae(e,[1,n[2],n[0],n[1]]);if(t==="channelsLast")return n.length===1?ae(e,[1,1,1,n[0]]):ae(e,[1].concat(n))}else if(s===3){if(t==="channelsFirst")return n.length===1?ae(e,[1,n[0],1]):ae(e,[1,n[1],n[0]]);if(t==="channelsLast")return n.length===1?ae(e,[1,1,n[0]]):ae(e,[1].concat(n))}else if(s<3)return e;throw new q(`Unsupported input rank by biasAdd: ${e.rank}`)}function Po(s,e,t){return Q(()=>(t==null&&(t=ki()),Ze(t),he(s,zv(s.rank,e,t))))}function Wv(s,e=1){if(e!==1)throw new Se(`Support for alpha values other than 1 (${e}) is not implemented yet.`);return Np(s)}function Gv(s){return Q(()=>_e(s,he(Tt(s),1)))}function Vv(s){return Q(()=>{const e=he(.5,ne(.2,s));return An(e,0,1)})}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */class mt extends vi{getConfig(){return{}}}class am extends mt{apply(e,t=1){return Wv(e,t)}}am.className="elu";le(am);class lm extends mt{apply(e){return T2(e)}}lm.className="selu";le(lm);class um extends mt{apply(e){return Do(e)}}um.className="relu";le(um);class cm extends mt{apply(e){return Q(()=>Ha(6,Do(e)))}}cm.className="relu6";le(cm);class hm extends mt{apply(e){return e}}hm.className="linear";le(hm);class fm extends mt{apply(e){return kc(e)}}fm.className="sigmoid";le(fm);class dm extends mt{apply(e){return Vv(e)}}dm.className="hardSigmoid";le(dm);class pm extends mt{apply(e){return Ac(e)}}pm.className="softplus";le(pm);class mm extends mt{apply(e){return Gv(e)}}mm.className="softsign";le(mm);class gm extends mt{apply(e){return Ic(e)}}gm.className="tanh";le(gm);class ym extends mt{apply(e,t=-1){return Wp(e,t)}}ym.className="softmax";le(ym);class bm extends mt{apply(e,t=-1){return Lx(e,t)}}bm.className="logSoftmax";le(bm);class wm extends mt{apply(e){return Q(()=>Q(()=>{const t=Math.sqrt(2),n=ne(.5,he(1,ex(_e(e,t))));return ne(e,n)}))}}wm.className="gelu";le(wm);class xm extends mt{apply(e){return Q(()=>ne(.5,ne(e,he(1,Ic(ne(Cn(_e(2,Math.PI)),he(e,ne(.044715,qa(e,3)))))))))}}xm.className="gelu_new";le(xm);class _m extends mt{apply(e){return Q(()=>ne(e,Ic(Ac(e))))}}_m.className="mish";le(_m);class vm extends mt{apply(e,t=1){return Q(()=>ne(kc(ne(e,t)),e))}}vm.className="swish";le(vm);function qv(s){return s.getClassName()}function ou(s,e={}){return Oo(s,on.getMap().classNameMap,e,"activation")}function Hv(s){if(s==null){const e={};return e.className="linear",e.config={},ou(e)}if(typeof s=="string"){const e={};return e.className=s,e.config={},ou(e)}else return s instanceof mt?s:ou(s)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function Bc(s,e){return Q(()=>Cn(Ae(ne(s,s),e,!0)))}class Ro extends vi{getConfig(){return{}}}class Sm extends Ro{constructor(e){super(),this.defaultMaxValue=2,this.defaultAxis=0,this.maxValue=e.maxValue!=null?e.maxValue:this.defaultMaxValue,this.axis=e.axis!=null?e.axis:this.defaultAxis}apply(e){return Q(()=>{const t=Bc(e,this.axis),n=An(t,0,this.maxValue);return ne(e,_e(n,he(Ye(),t)))})}getConfig(){return{maxValue:this.maxValue,axis:this.axis}}}Sm.className="MaxNorm";le(Sm);class km extends Ro{constructor(e){super(),this.defaultAxis=0,this.axis=e.axis!=null?e.axis:this.defaultAxis}apply(e){return Q(()=>_e(e,he(Ye(),Bc(e,this.axis))))}getConfig(){return{axis:this.axis}}}km.className="UnitNorm";le(km);class Im extends Ro{apply(e){return Do(e)}}Im.className="NonNeg";le(Im);class Em extends Ro{constructor(e){super(),this.defaultMinValue=0,this.defaultMaxValue=1,this.defaultRate=1,this.defaultAxis=0,this.minValue=e.minValue!=null?e.minValue:this.defaultMinValue,this.maxValue=e.maxValue!=null?e.maxValue:this.defaultMaxValue,this.rate=e.rate!=null?e.rate:this.defaultRate,this.axis=e.axis!=null?e.axis:this.defaultAxis}apply(e){return Q(()=>{const t=Bc(e,this.axis),n=he(ne(this.rate,An(t,this.minValue,this.maxValue)),ne(1-this.rate,t));return ne(e,_e(n,he(Ye(),t)))})}getConfig(){return{minValue:this.minValue,maxValue:this.maxValue,rate:this.rate,axis:this.axis}}}Em.className="MinMaxNorm";le(Em);const df={maxNorm:"MaxNorm",minMaxNorm:"MinMaxNorm",nonNeg:"NonNeg",unitNorm:"UnitNorm"};function Za(s){return Pc(s)}function pf(s,e={}){return Oo(s,on.getMap().classNameMap,e,"constraint")}function Qa(s){if(s==null)return null;if(typeof s=="string"){const t={className:s in df?df[s]:s,config:{}};return pf(t)}else return s instanceof Ro?s:pf(s)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */let jv=0;function Tm(){return jv++}const Jo={};function Fc(s=""){return s in Jo||(Jo[s]=0),Jo[s]+=1,s+Jo[s].toString()}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */const Kv=["fanIn","fanOut","fanAvg"],Xv=["normal","uniform","truncatedNormal"];/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function Yv(s){Si(Kv,"FanMode",s)}function Zv(s){Si(Xv,"Distribution",s)}class ns extends vi{fromConfigUsesCustomObjects(){return!1}getConfig(){return{}}}class Am extends ns{apply(e,t){return fi(e,t)}}Am.className="Zeros";le(Am);class Cm extends ns{apply(e,t){return Cc(e,t)}}Cm.className="Ones";le(Cm);class Nm extends ns{constructor(e){if(super(),typeof e!="object")throw new q(`Expected argument of type ConstantConfig but got ${e}`);if(e.value===void 0)throw new q(`config must have value set but got ${e}`);this.value=e.value}apply(e,t){return Q(()=>ne(Qt(this.value),Cc(e,t)))}getConfig(){return{value:this.value}}}Nm.className="Constant";le(Nm);class $m extends ns{constructor(e){super(),this.DEFAULT_MINVAL=-.05,this.DEFAULT_MAXVAL=.05,this.minval=e.minval||this.DEFAULT_MINVAL,this.maxval=e.maxval||this.DEFAULT_MAXVAL,this.seed=e.seed}apply(e,t){return Up(e,this.minval,this.maxval,t,this.seed)}getConfig(){return{minval:this.minval,maxval:this.maxval,seed:this.seed}}}$m.className="RandomUniform";le($m);class Dm extends ns{constructor(e){super(),this.DEFAULT_MEAN=0,this.DEFAULT_STDDEV=.05,this.mean=e.mean||this.DEFAULT_MEAN,this.stddev=e.stddev||this.DEFAULT_STDDEV,this.seed=e.seed}apply(e,t){if(t=t||"float32",t!=="float32"&&t!=="int32")throw new Se(`randomNormal does not support dType ${t}.`);return om(e,this.mean,this.stddev,t,this.seed)}getConfig(){return{mean:this.mean,stddev:this.stddev,seed:this.seed}}}Dm.className="RandomNormal";le(Dm);class Om extends ns{constructor(e){super(),this.DEFAULT_MEAN=0,this.DEFAULT_STDDEV=.05,this.mean=e.mean||this.DEFAULT_MEAN,this.stddev=e.stddev||this.DEFAULT_STDDEV,this.seed=e.seed}apply(e,t){if(t=t||"float32",t!=="float32"&&t!=="int32")throw new Se(`truncatedNormal does not support dType ${t}.`);return Vp(e,this.mean,this.stddev,t,this.seed)}getConfig(){return{mean:this.mean,stddev:this.stddev,seed:this.seed}}}Om.className="TruncatedNormal";le(Om);class Mm extends ns{constructor(e){super(),this.gain=e.gain!=null?e.gain:1}apply(e,t){return Q(()=>{if(e.length!==2||e[0]!==e[1])throw new q("Identity matrix initializer can only be used for 2D square matrices.");return ne(this.gain,Pp(e[0]))})}getConfig(){return{gain:this.gain}}}Mm.className="Identity";le(Mm);function Qv(s,e="channelsLast"){let t,n;if(Ze(e),s.length===2)t=s[0],n=s[1];else if([3,4,5].indexOf(s.length)!==-1){if(e==="channelsFirst"){const r=no(s,2);t=s[1]*r,n=s[0]*r}else if(e==="channelsLast"){const r=no(s,0,s.length-2);t=s[s.length-2]*r,n=s[s.length-1]*r}}else{const r=no(s);t=Math.sqrt(r),n=Math.sqrt(r)}return[t,n]}class Ut extends ns{constructor(e){if(super(),e.scale<0)throw new q(`scale must be a positive float. Got: ${e.scale}`);this.scale=e.scale==null?1:e.scale,this.mode=e.mode==null?"fanIn":e.mode,Yv(this.mode),this.distribution=e.distribution==null?"normal":e.distribution,Zv(this.distribution),this.seed=e.seed}apply(e,t){const n=Qv(e),r=n[0],i=n[1];let o=this.scale;if(this.mode==="fanIn"?o/=Math.max(1,r):this.mode==="fanOut"?o/=Math.max(1,i):o/=Math.max(1,(r+i)/2),this.distribution==="normal"){const a=Math.sqrt(o);if(t=t||"float32",t!=="float32"&&t!=="int32")throw new Se(`${this.getClassName()} does not support dType ${t}.`);return Vp(e,0,a,t,this.seed)}else{const a=Math.sqrt(3*o);return Up(e,-a,a,t,this.seed)}}getConfig(){return{scale:this.scale,mode:this.mode,distribution:this.distribution,seed:this.seed}}}Ut.className="VarianceScaling";le(Ut);class Uc extends Ut{constructor(e){super({scale:1,mode:"fanAvg",distribution:"uniform",seed:e==null?null:e.seed})}getClassName(){return Ut.className}}Uc.className="GlorotUniform";le(Uc);class zc extends Ut{constructor(e){super({scale:1,mode:"fanAvg",distribution:"normal",seed:e==null?null:e.seed})}getClassName(){return Ut.className}}zc.className="GlorotNormal";le(zc);class Wc extends Ut{constructor(e){super({scale:2,mode:"fanIn",distribution:"normal",seed:e==null?null:e.seed})}getClassName(){return Ut.className}}Wc.className="HeNormal";le(Wc);class Gc extends Ut{constructor(e){super({scale:2,mode:"fanIn",distribution:"uniform",seed:e==null?null:e.seed})}getClassName(){return Ut.className}}Gc.className="HeUniform";le(Gc);class Vc extends Ut{constructor(e){super({scale:1,mode:"fanIn",distribution:"normal",seed:e==null?null:e.seed})}getClassName(){return Ut.className}}Vc.className="LeCunNormal";le(Vc);class qc extends Ut{constructor(e){super({scale:1,mode:"fanIn",distribution:"uniform",seed:e==null?null:e.seed})}getClassName(){return Ut.className}}qc.className="LeCunUniform";le(qc);class Pm extends ns{constructor(e){super(),this.DEFAULT_GAIN=1,this.ELEMENTS_WARN_SLOW=2e3,this.gain=e.gain==null?this.DEFAULT_GAIN:e.gain,this.seed=e.seed}apply(e,t){return Q(()=>{if(e.length<2)throw new Se("Shape must be at least 2D.");if(t!=="int32"&&t!=="float32"&&t!==void 0)throw new TypeError(`Unsupported data type ${t}.`);t=t;const n=me(e.slice(0,-1)),r=e[e.length-1],i=n*r;i>this.ELEMENTS_WARN_SLOW&&console.warn(`Orthogonal initializer is being called on a matrix with more than ${this.ELEMENTS_WARN_SLOW} (${i}) elements: Slowness may result.`);const o=[Math.max(r,n),Math.min(r,n)],a=om(o,0,1,t,this.seed),l=G_.qr(a,!1);let u=l[0];const h=l[1].flatten().stridedSlice([0],[Math.min(r,n)*Math.min(r,n)],[Math.min(r,n)+1]);return u=ne(u,h.sign()),n<r&&(u=u.transpose()),ne(Qt(this.gain),u.reshape(e))})}getConfig(){return{gain:this.gain,seed:this.seed}}}Pm.className="Orthogonal";le(Pm);const mf={constant:"Constant",glorotNormal:"GlorotNormal",glorotUniform:"GlorotUniform",heNormal:"HeNormal",heUniform:"HeUniform",identity:"Identity",leCunNormal:"LeCunNormal",leCunUniform:"LeCunUniform",ones:"Ones",orthogonal:"Orthogonal",randomNormal:"RandomNormal",randomUniform:"RandomUniform",truncatedNormal:"TruncatedNormal",varianceScaling:"VarianceScaling",zeros:"Zeros"};function gf(s,e={}){return Oo(s,on.getMap().classNameMap,e,"initializer")}function Ja(s){return Pc(s)}function _o(s){if(typeof s=="string"){const e=s in mf?mf[s]:s;if(e==="GlorotNormal")return new zc;if(e==="GlorotUniform")return new Uc;if(e==="HeNormal")return new Wc;if(e==="HeUniform")return new Gc;if(e==="LeCunNormal")return new Vc;if(e==="LeCunUniform")return new qc;{const t={};return t.className=e,t.config={},gf(t)}}else return s instanceof ns?s:gf(s)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function el(s){return s.length===0?[]:Array.isArray(s[0])?s:[s]}function zt(s){let e;if(Array.isArray(s)){if(s.length!==1)throw new q(`Expected Tensor length to be 1; got ${s.length}`);e=s[0]}else e=s;return e}function Nn(s){if(Array.isArray(s)&&Array.isArray(s[0])){if(s.length===1)return s=s,s[0];throw new q(`Expected exactly 1 Shape; got ${s.length}`)}else return s}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function tl(s){let e=0;for(const t of s)t.shape.length===0?e+=1:e+=t.shape.reduce((n,r)=>n*r);return e}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */const yf="Variable";class Jv{constructor(e,t="float32",n=yf,r=!0,i=null){this.dtype=t??"float32",this.shape=e.shape,this.id=Tm(),n=n??yf,this.originalName=tm(n),this.name=nm(this.originalName),this.trainable_=r,this.constraint=i,this.val=z2(e,this.trainable_,this.name,this.dtype)}read(){return this.assertNotDisposed(),this.val}write(e){return this.assertNotDisposed(),e3(this.val,e),this.val.id!==e.id&&(this.val.assign(e),this.constraint!=null&&this.val.assign(this.constraint.apply(this.val))),this}dispose(){this.assertNotDisposed(),this.val.dispose()}assertNotDisposed(){if(this.val.isDisposed)throw new Error(`LayersVariable ${this.name} is already disposed.`)}get trainable(){return this.trainable_}set trainable(e){this.trainable_=e,this.val.trainable=e}}function e3(s,e){if(s.shape.toString()!==e.shape.toString())throw new Error("Shape mismatch: "+JSON.stringify(s.shape)+" vs. "+JSON.stringify(e.shape))}function bf(s){return s.map(e=>e.read())}function Rm(s){s.forEach(e=>{e[0].write(e[1])})}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */class Jn{constructor(e){this.dtype=e.dtype,this.shape=e.shape,e.shape!=null?this.ndim=e.shape.length:this.ndim=e.ndim,this.maxNDim=e.maxNDim,this.minNDim=e.minNDim,this.axes=e.axes||{}}}class Er{constructor(e,t,n,r,i,o,a){this.dtype=e,this.shape=t,this.sourceLayer=n,this.inputs=r,this.callArgs=i,this.outputTensorIndex=a,this.id=Tm(),o!=null&&(this.originalName=tm(o),this.name=nm(this.originalName)),this.rank=t.length}}let t3=0;class Hc{constructor(e,t){this.callArgs=t,this.id=t3++,this.outboundLayer=e.outboundLayer,this.inboundLayers=e.inboundLayers,this.nodeIndices=e.nodeIndices,this.tensorIndices=e.tensorIndices,this.inputTensors=e.inputTensors,this.outputTensors=e.outputTensors,this.inputMasks=e.inputMasks,this.outputMasks=e.outputMasks,this.inputShapes=e.inputShapes,this.outputShapes=e.outputShapes;for(const n of e.inboundLayers)n?.outboundNodes.push(this);e.outboundLayer.inboundNodes.push(this)}getConfig(){const e=[];for(const t of this.inboundLayers)t!=null?e.push(t.name):e.push(null);return{outboundLayer:this.outboundLayer?this.outboundLayer.name:null,inboundLayers:e,nodeIndices:this.nodeIndices,tensorIndices:this.tensorIndices}}}let n3=0;class $n extends vi{constructor(e={}){super(),this._callHook=null,this._addedWeightNames=[],this._stateful=!1,this.id=n3++,this.activityRegularizer=null,this.inputSpec=null,this.supportsMasking=!1,this._trainableWeights=[],this._nonTrainableWeights=[],this._losses=[],this._updates=[],this._built=!1,this.inboundNodes=[],this.outboundNodes=[];let t=e.name;if(!t){const n=this.getClassName();t=us(n)+"_"+Fc(n)}if(this.name=t,this.trainable_=e.trainable==null?!0:e.trainable,e.inputShape!=null||e.batchInputShape!=null){let n;if(e.batchInputShape!=null)n=e.batchInputShape;else if(e.inputShape!=null){let i=null;e.batchSize!=null&&(i=e.batchSize),n=[i].concat(e.inputShape)}this.batchInputShape=n;let r=e.dtype;r==null&&(r=e.inputDType),r==null&&(r="float32"),this.dtype=r}e.weights!=null?this.initialWeights=e.weights:this.initialWeights=null,this._refCount=null,this.fastWeightInitDuringBuild=!1}static nodeKey(e,t){return e.name+"_ib-"+t.toString()}getNodeAtIndex(e,t){if(this.inboundNodes.length===0)throw new Ms(`The layer has never been called and thus has no defined ${t}.`);if(this.inboundNodes.length<=e)throw new q(`Asked to get ${t} at node ${e}, but the layer has only ${this.inboundNodes.length} inbound nodes.`);return this.inboundNodes[e]}getInputAt(e){return Rt(this.getNodeAtIndex(e,"input").inputTensors)}getOutputAt(e){return Rt(this.getNodeAtIndex(e,"output").outputTensors)}get input(){if(this.inboundNodes.length>1)throw new ks(`Layer ${this.name} has multiple inbound nodes, hence the notion of "layer input" is ill-defined. Use \`getInputAt(nodeIndex)\` instead.`);if(this.inboundNodes.length===0)throw new ks(`Layer ${this.name} is not connected, no input to return.`);return Rt(this.getNodeAtIndex(0,"input").inputTensors)}get output(){if(this.inboundNodes.length===0)throw new ks(`Layer ${this.name} has no inbound nodes.`);if(this.inboundNodes.length>1)throw new ks(`Layer ${this.name} has multiple inbound nodes, hence the notion of "layer output" is ill-defined. Use \`getOutputAt(nodeIndex)\` instead.`);return Rt(this.getNodeAtIndex(0,"output").outputTensors)}get losses(){return this._losses}calculateLosses(){return this.losses.map(e=>e())}get updates(){return this._updates}get built(){return this._built}set built(e){this._built=e}get trainable(){return this.trainable_}set trainable(e){this._trainableWeights.forEach(t=>t.trainable=e),this.trainable_=e}get trainableWeights(){return this.trainable_?this._trainableWeights.filter(e=>e.trainable):[]}set trainableWeights(e){this._trainableWeights=e}get nonTrainableWeights(){return this.trainable?this._trainableWeights.filter(e=>!e.trainable).concat(this._nonTrainableWeights):this._trainableWeights.concat(this._nonTrainableWeights)}set nonTrainableWeights(e){this._nonTrainableWeights=e}get weights(){return this.trainableWeights.concat(this.nonTrainableWeights)}get stateful(){return this._stateful}resetStates(){if(!this.stateful)throw new Error("Cannot call the resetStates() method of a non-stateful Layer object.")}assertInputCompatibility(e){const t=Ne(e);if(this.inputSpec==null||this.inputSpec.length===0)return;const n=Ne(this.inputSpec);if(t.length!==n.length)throw new q(`Layer ${this.name} expects ${n.length} inputs, but it received ${t.length} input tensors. Input received: ${e}`);for(let r=0;r<t.length;r++){const i=t[r],o=n[r];if(o==null)continue;const a=i.rank;if(o.ndim!=null&&a!==o.ndim)throw new q(`Input ${r} is incompatible with layer ${this.name}: expected ndim=${o.ndim}, found ndim=${a}`);if(o.maxNDim!=null&&a>o.maxNDim)throw new q(`Input ${r} is incompatible with layer ${this.name}: expected max_ndim=${o.maxNDim}, found ndim=${a}`);if(o.minNDim!=null&&a<o.minNDim)throw new q(`Input ${r} is incompatible with layer ${this.name}: expected min_ndim=${o.minNDim}, found ndim=${a}.`);if(o.dtype!=null&&i.dtype!==o.dtype)throw new q(`Input ${r} is incompatible with layer ${this.name} : expected dtype=${o.dtype}, found dtype=${i.dtype}.`);if(o.axes){const l=i.shape;for(const u in o.axes){const c=Number(u),h=o.axes[u],d=c>=0?l[c]:l[l.length+c];if(h!=null&&[h,null].indexOf(d)===-1)throw new q(`Input ${r} is incompatible with layer ${this.name}: expected axis ${c} of input shape to have value ${h} but got shape ${l}.`)}}if(o.shape!=null)for(let l=0;l<o.shape.length;++l){const u=o.shape[l],c=i.shape[l];if(u!=null&&c!=null&&u!==c)throw new q(`Input ${r} is incompatible with layer ${this.name}: expected shape=${o.shape}, found shape=${i.shape}.`)}}}call(e,t){return e}invokeCallHook(e,t){this._callHook!=null&&this._callHook(e,t)}setCallHook(e){this._callHook=e}clearCallHook(){this._callHook=null}apply(e,t){t=t||{},this.assertNotDisposed();const n=Ne(e),r=i3(e),i=o3(e);if(r===i)throw new q("Arguments to apply() must be all SymbolicTensors or all Tensors");return Ia(this.name,()=>{if(!this.built){this.assertInputCompatibility(e);const o=[];for(const a of Ne(e))o.push(a.shape);this.build(Rt(o)),this.built=!0,this.initialWeights&&this.setWeights(this.initialWeights),this._refCount===null&&i&&(this._refCount=1)}if(this.assertInputCompatibility(e),i){let o=this.call(e,t);this.supportsMasking&&this.setMaskMetadata(e,o);const a=Ne(o),l=[];for(let u of a)n.indexOf(u)!==-1&&(u=u.clone()),l.push(u);if(o=Rt(l),this.activityRegularizer!=null)throw new Se("Layer invocation in the presence of activity regularizer(s) is not supported yet.");return o}else{const o=s3(e),a=this.computeOutputShape(o);let l;const u=r3(e);if(this.warnOnIncompatibleInputShape(Array.isArray(e)?o[0]:o),a!=null&&a.length>0&&Array.isArray(a[0])?l=a.map((c,h)=>new Er(u,c,this,Ne(e),t,this.name,h)):l=new Er(u,a,this,Ne(e),t,this.name),this.addInboundNode(e,l,null,null,o,a,t),this._refCount++,this.activityRegularizer!=null)throw new Se("Layer invocation in the presence of activity regularizer(s) is not supported yet.");return l}})}warnOnIncompatibleInputShape(e){if(this.batchInputShape!=null)if(e.length!==this.batchInputShape.length)console.warn(`The rank of the input tensor provided (shape: ${JSON.stringify(e)}) does not match that of the batchInputShape (${JSON.stringify(this.batchInputShape)}) of the layer ${this.name}`);else{let t=!1;this.batchInputShape.forEach((n,r)=>{n!=null&&e[r]!=null&&e[r]!==n&&(t=!0)}),t&&console.warn(`The shape of the input tensor (${JSON.stringify(e)}) does not match the expectation of layer ${this.name}: ${JSON.stringify(this.batchInputShape)}`)}}get outputShape(){if(this.inboundNodes==null||this.inboundNodes.length===0)throw new ks(`The layer ${this.name} has never been called and thus has no defined output shape.`);const e=[];for(const t of this.inboundNodes){const n=JSON.stringify(t.outputShapes);e.indexOf(n)===-1&&e.push(n)}if(e.length===1){const t=this.inboundNodes[0].outputShapes;return Array.isArray(t)&&Array.isArray(t[0])&&t.length===1?t[0]:t}else throw new ks(`The layer ${this.name} has multiple inbound nodes with different output shapes. Hence the notion of "output shape" is ill-defined for the layer.`)}countParams(){if(!this.built)throw new Ms(`You tried to call countParams() on ${this.name}, but the layer is not built yet. Build it first by calling build(batchInputShape).`);return tl(this.weights)}build(e){this.built=!0}getWeights(e=!1){return bf(e?this.trainableWeights:this.weights)}setWeights(e){Q(()=>{const t=this.weights;if(t.length!==e.length)throw new q(`You called setWeights(weights) on layer "${this.name}" with a weight list of length ${e.length}, but the layer was expecting ${t.length} weights. Provided weights: ${e}...`);if(t.length===0)return;const n=[],r=bf(t);for(let i=0;i<r.length;++i){const o=r[i],a=t[i],l=e[i];if(!cn(o.shape,l.shape))throw new q(`Layer weight shape ${o.shape} not compatible with provided weight shape ${l.shape}`);n.push([a,l])}Rm(n)})}addWeight(e,t,n,r,i,o,a,l){if(this._addedWeightNames.indexOf(e)!==-1)throw new q(`Duplicate weight name ${e} for layer ${this.name}`);this._addedWeightNames.push(e),n==null&&(n="float32"),this.fastWeightInitDuringBuild&&(r=l!=null?l():_o("zeros"));const u=r.apply(t,n),c=new Jv(u,n,e,o,a);return u.dispose(),i!=null&&this.addLoss(()=>i.apply(c.read())),o==null&&(o=!0),o?this._trainableWeights.push(c):this._nonTrainableWeights.push(c),c}setFastWeightInitDuringBuild(e){this.fastWeightInitDuringBuild=e}addLoss(e){e==null||Array.isArray(e)&&e.length===0||(e=Ne(e),this._losses!==void 0&&this._losses!==null&&this.losses.push(...e))}computeOutputShape(e){return e}computeMask(e,t){if(!this.supportsMasking){if(t!=null)if(Array.isArray(t))t.forEach(n=>{if(n!=null)throw new TypeError(`Layer ${this.name} does not support masking, but was passed an inputMask.`)});else throw new TypeError(`Layer ${this.name} does not support masking, but was passed an inputMask.`);return null}return t}setMaskMetadata(e,t,n){if(!this.supportsMasking)return;const r=this.computeMask(e,n),i=Ne(t),o=Ne(r);if(i.length!==o.length)throw new Error(`${this.name} outputs ${i.length} tensors but ${i.length} masks for those tensors`);for(let a=0;a<i.length;a++)i[a].kerasMask=o[a]}addInboundNode(e,t,n,r,i,o,a=null){const l=Ne(e);t=Ne(t),n=Ne(n),r=Ne(r),i=el(i),o=el(o);const u=[],c=[],h=[];for(const d of l)u.push(d.sourceLayer),c.push(d.nodeIndex),h.push(d.tensorIndex);new Hc({outboundLayer:this,inboundLayers:u,nodeIndices:c,tensorIndices:h,inputTensors:l,outputTensors:t,inputMasks:n,outputMasks:r,inputShapes:i,outputShapes:o},a);for(let d=0;d<t.length;d++)t[d].sourceLayer=this,t[d].nodeIndex=this.inboundNodes.length-1,t[d].tensorIndex=d}getConfig(){const e={name:this.name,trainable:this.trainable};return this.batchInputShape!=null&&(e.batchInputShape=this.batchInputShape),this.dtype!=null&&(e.dtype=this.dtype),e}disposeWeights(){return this.weights.forEach(e=>e.dispose()),this.weights.length}assertNotDisposed(){if(this._refCount===0)throw new Error(`Layer '${this.name}' is already disposed.`)}dispose(){if(!this.built)throw new Error(`Cannot dispose Layer ${this.name} because it has not been built yet.`);if(this._refCount===null)throw new Error(`Cannot dispose Layer ${this.name} because it has not been used yet.`);this.assertNotDisposed();let e=0;return--this._refCount===0&&(e=this.disposeWeights()),{refCountAfterDispose:this._refCount,numDisposedVariables:e}}}function s3(s){s=Ne(s);const e=[];for(const t of s)e.push(t.shape);return Rt(e)}function r3(s){return"float32"}function i3(s){let e=!0;for(const t of Ne(s))if(!(t instanceof Er)){e=!1;break}return e}function o3(s){let e=!0;for(const t of Ne(s))if(t instanceof Er){e=!1;break}return e}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function a3(s){if(s!=null&&typeof s!="object")throw new Error(`Argument to L1L2 regularizer's constructor is expected to be an object, but received: ${s}`)}class Lm extends vi{}class Bm extends Lm{constructor(e){super(),a3(e),this.l1=e==null||e.l1==null?.01:e.l1,this.l2=e==null||e.l2==null?.01:e.l2,this.hasL1=this.l1!==0,this.hasL2=this.l2!==0}apply(e){return Q(()=>{let t=fi([1]);return this.hasL1&&(t=he(t,Ae(ne(this.l1,Tt(e))))),this.hasL2&&(t=he(t,Ae(ne(this.l2,Mo(e))))),ae(t,[])})}getConfig(){return{l1:this.l1,l2:this.l2}}static fromConfig(e,t){return new e({l1:t.l1,l2:t.l2})}}Bm.className="L1L2";le(Bm);const wf={l1l2:"L1L2"};function vo(s){return Pc(s)}function xf(s,e={}){return Oo(s,on.getMap().classNameMap,e,"regularizer")}function So(s){if(s==null)return null;if(typeof s=="string"){const t={className:s in wf?wf[s]:s,config:{}};return xf(t)}else return s instanceof Lm?s:xf(s)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function au(s,e,t){if(typeof s=="number")return Xa(s,e);if(s.length!==e)throw new q(`The ${t} argument must be an integer or tuple of ${e} integers. Received: ${s.length} elements.`);for(let n=0;n<e;++n){const r=s[n];if(!Lv(r))throw new q(`The ${t} argument must be an integer or tuple of ${e} integers. Received: ${JSON.stringify(s)} including a non-integer number ${r}`)}return s}function br(s,e,t,n,r=1){if(s==null)return s;const i=e+(e-1)*(r-1);let o;return t==="same"?o=s:o=s-i+1,Math.floor((o+n-1)/n)}function Yn(s,e,t,n){if(s==null)return null;if(n==="valid")s=s*e+rm([t-e,0]);else if(n==="same")s=s*e;else throw new q(`Unsupport padding mode: ${n}.`);return s}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function Fm(s,e){return Q(()=>(Ze(e),e==="channelsFirst"?He(s,[0,2,3,1]):s))}function Um(s,e){return Q(()=>(Ze(e),e==="channelsFirst"?He(s,[0,2,3,4,1]):s))}function l3(s,e,t,n=1,r="valid",i,o=1){return Q(()=>{if(i==null&&(i=ki()),Ze(i),s.shape.length!==3)throw new q(`The input of a conv1dWithBias operation should be 3, but is ${s.shape.length} instead.`);if(e.shape.length!==3)throw new q(`The kernel for a conv1dWithBias operation should be 3, but is ${e.shape.length} instead`);if(t!=null&&t.shape.length!==1)throw new q(`The bias for a conv1dWithBias operation should be 1, but is ${t.shape.length} instead`);if(i==="channelsFirst"&&(s=He(s,[0,2,1])),r==="causal")throw new Se("The support for CAUSAL padding mode in conv1dWithBias is not implemented yet.");let a=Pw(s,e,n,r==="same"?"same":"valid","NWC",o);return t!=null&&(a=Po(a,t)),a})}function _f(s,e,t,n=[1,1],r="valid",i,o,a=null){return Q(()=>{if(i==null&&(i=ki()),Ze(i),s.rank!==3&&s.rank!==4)throw new q(`conv2dWithBiasActivation expects input to be of rank 3 or 4, but received ${s.rank}.`);if(e.rank!==3&&e.rank!==4)throw new q(`conv2dWithBiasActivation expects kernel to be of rank 3 or 4, but received ${s.rank}.`);let l=Fm(s,i);if(r==="causal")throw new Se("The support for CAUSAL padding mode in conv1dWithBias is not implemented yet.");return l=Y2({x:l,filter:e,strides:n,pad:r==="same"?"same":"valid",dilations:o,dataFormat:"NHWC",bias:t,activation:a}),i==="channelsFirst"&&(l=He(l,[0,3,1,2])),l})}function u3(s,e,t,n=[1,1,1],r="valid",i,o){return Q(()=>{if(i==null&&(i=ki()),Ze(i),s.rank!==4&&s.rank!==5)throw new q(`conv3dWithBias expects input to be of rank 4 or 5, but received ${s.rank}.`);if(e.rank!==4&&e.rank!==5)throw new q(`conv3dWithBias expects kernel to be of rank 4 or 5, but received ${s.rank}.`);let a=Um(s,i);if(r==="causal")throw new Se("The support for CAUSAL padding mode in conv3dWithBias is not implemented yet.");return a=Uw(a,e,n,r==="same"?"same":"valid","NDHWC",o),t!=null&&(a=Po(a,t)),i==="channelsFirst"&&(a=He(a,[0,4,1,2,3])),a})}class jc extends $n{constructor(e,t){if(super(t),this.bias=null,this.DEFAULT_KERNEL_INITIALIZER="glorotNormal",this.DEFAULT_BIAS_INITIALIZER="zeros",jc.verifyArgs(t),this.rank=e,ys(this.rank,"rank"),this.rank!==1&&this.rank!==2&&this.rank!==3)throw new Se(`Convolution layer for rank other than 1, 2, or 3 (${this.rank}) is not implemented yet.`);if(this.kernelSize=au(t.kernelSize,e,"kernelSize"),this.strides=au(t.strides==null?1:t.strides,e,"strides"),this.padding=t.padding==null?"valid":t.padding,hn(this.padding),this.dataFormat=t.dataFormat==null?"channelsLast":t.dataFormat,Ze(this.dataFormat),this.activation=Hv(t.activation),this.useBias=t.useBias==null?!0:t.useBias,this.biasInitializer=_o(t.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.biasConstraint=Qa(t.biasConstraint),this.biasRegularizer=So(t.biasRegularizer),this.activityRegularizer=So(t.activityRegularizer),this.dilationRate=au(t.dilationRate==null?1:t.dilationRate,e,"dilationRate"),this.rank===1&&Array.isArray(this.dilationRate)&&this.dilationRate.length!==1)throw new q(`dilationRate must be a number or an array of a single number for 1D convolution, but received ${JSON.stringify(this.dilationRate)}`);if(this.rank===2){if(typeof this.dilationRate=="number")this.dilationRate=[this.dilationRate,this.dilationRate];else if(this.dilationRate.length!==2)throw new q(`dilationRate must be a number or array of two numbers for 2D convolution, but received ${JSON.stringify(this.dilationRate)}`)}else if(this.rank===3){if(typeof this.dilationRate=="number")this.dilationRate=[this.dilationRate,this.dilationRate,this.dilationRate];else if(this.dilationRate.length!==3)throw new q(`dilationRate must be a number or array of three numbers for 3D convolution, but received ${JSON.stringify(this.dilationRate)}`)}}static verifyArgs(e){if(Xn("kernelSize"in e,"required key 'kernelSize' not in config"),typeof e.kernelSize!="number"&&!Rc(e.kernelSize,"number",1,3))throw new q(`BaseConv expects config.kernelSize to be number or number[] with length 1, 2, or 3, but received ${JSON.stringify(e.kernelSize)}.`)}getConfig(){const e={kernelSize:this.kernelSize,strides:this.strides,padding:this.padding,dataFormat:this.dataFormat,dilationRate:this.dilationRate,activation:qv(this.activation),useBias:this.useBias,biasInitializer:Ja(this.biasInitializer),biasRegularizer:vo(this.biasRegularizer),activityRegularizer:vo(this.activityRegularizer),biasConstraint:Za(this.biasConstraint)},t=super.getConfig();return Object.assign(e,t),e}}class Ii extends jc{constructor(e,t){super(e,t),this.kernel=null,Ii.verifyArgs(t),this.filters=t.filters,ys(this.filters,"filters"),this.kernelInitializer=_o(t.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.kernelConstraint=Qa(t.kernelConstraint),this.kernelRegularizer=So(t.kernelRegularizer)}build(e){e=Nn(e);const t=this.dataFormat==="channelsFirst"?1:e.length-1;if(e[t]==null)throw new q(`The channel dimension of the input should be defined. Found ${e[t]}`);const n=e[t],r=this.kernelSize.concat([n,this.filters]);this.kernel=this.addWeight("kernel",r,null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight("bias",[this.filters],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint)),this.inputSpec=[{ndim:this.rank+2,axes:{[t]:n}}],this.built=!0}call(e,t){return Q(()=>{e=zt(e);let n;const r=this.bias==null?null:this.bias.read(),i=Ov(this.activation.getClassName());if(i!=null&&this.rank===2)n=_f(e,this.kernel.read(),r,this.strides,this.padding,this.dataFormat,this.dilationRate,i);else{if(this.rank===1)n=l3(e,this.kernel.read(),r,this.strides[0],this.padding,this.dataFormat,this.dilationRate[0]);else if(this.rank===2)n=_f(e,this.kernel.read(),r,this.strides,this.padding,this.dataFormat,this.dilationRate);else if(this.rank===3)n=u3(e,this.kernel.read(),r,this.strides,this.padding,this.dataFormat,this.dilationRate);else throw new Se("convolutions greater than 3D are not implemented yet.");this.activation!=null&&(n=this.activation.apply(n))}return n})}computeOutputShape(e){e=Nn(e);const t=[],n=this.dataFormat==="channelsLast"?e.slice(1,e.length-1):e.slice(2);for(let i=0;i<n.length;++i){const o=br(n[i],this.kernelSize[i],this.padding,this.strides[i],typeof this.dilationRate=="number"?this.dilationRate:this.dilationRate[i]);t.push(o)}let r=[e[0]];return this.dataFormat==="channelsLast"?(r=r.concat(t),r.push(this.filters)):(r.push(this.filters),r=r.concat(t)),r}getConfig(){const e={filters:this.filters,kernelInitializer:Ja(this.kernelInitializer),kernelRegularizer:vo(this.kernelRegularizer),kernelConstraint:Za(this.kernelConstraint)},t=super.getConfig();return Object.assign(e,t),e}static verifyArgs(e){if(!("filters"in e)||typeof e.filters!="number"||e.filters<1)throw new q(`Convolution layer expected config.filters to be a 'number' > 0 but got ${JSON.stringify(e.filters)}`)}}class Ei extends Ii{constructor(e){super(2,e),Ei.verifyArgs(e)}getConfig(){const e=super.getConfig();return delete e.rank,e}static verifyArgs(e){if(typeof e.kernelSize!="number"&&!Rc(e.kernelSize,"number",1,2))throw new q(`Conv2D expects config.kernelSize to be number or number[] with length 1 or 2, but received ${JSON.stringify(e.kernelSize)}.`)}}Ei.className="Conv2D";le(Ei);class Lo extends Ii{constructor(e){super(3,e),Lo.verifyArgs(e)}getConfig(){const e=super.getConfig();return delete e.rank,e}static verifyArgs(e){if(typeof e.kernelSize!="number"&&!(Array.isArray(e.kernelSize)&&(e.kernelSize.length===1||e.kernelSize.length===3)))throw new q(`Conv3D expects config.kernelSize to be number or [number, number, number], but received ${JSON.stringify(e.kernelSize)}.`)}}Lo.className="Conv3D";le(Lo);class zm extends Ei{constructor(e){if(super(e),this.inputSpec=[new Jn({ndim:4})],this.padding!=="same"&&this.padding!=="valid")throw new q(`Conv2DTranspose currently supports only padding modes 'same' and 'valid', but received padding mode ${this.padding}`)}build(e){if(e=Nn(e),e.length!==4)throw new q("Input should have rank 4; Received input shape: "+JSON.stringify(e));const t=this.dataFormat==="channelsFirst"?1:e.length-1;if(e[t]==null)throw new q("The channel dimension of the inputs should be defined. Found `None`.");const n=e[t],r=this.kernelSize.concat([this.filters,n]);this.kernel=this.addWeight("kernel",r,"float32",this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight("bias",[this.filters],"float32",this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint)),this.inputSpec=[new Jn({ndim:4,axes:{[t]:n}})],this.built=!0}call(e,t){return Q(()=>{let n=zt(e);if(n.shape.length!==4)throw new q(`Conv2DTranspose.call() expects input tensor to be rank-4, but received a tensor of rank-${n.shape.length}`);const r=n.shape,i=r[0];let o,a;this.dataFormat==="channelsFirst"?(o=2,a=3):(o=1,a=2);const l=r[o],u=r[a],c=this.kernelSize[0],h=this.kernelSize[1],d=this.strides[0],w=this.strides[1],k=Yn(l,d,c,this.padding),A=Yn(u,w,h,this.padding),m=[i,k,A,this.filters];this.dataFormat!=="channelsLast"&&(n=He(n,[0,2,3,1]));let S=Bw(n,this.kernel.read(),m,this.strides,this.padding);return this.dataFormat!=="channelsLast"&&(S=He(S,[0,3,1,2])),this.bias!=null&&(S=Po(S,this.bias.read(),this.dataFormat)),this.activation!=null&&(S=this.activation.apply(S)),S})}computeOutputShape(e){e=Nn(e);const t=e.slice();let n,r,i;this.dataFormat==="channelsFirst"?(n=1,r=2,i=3):(n=3,r=1,i=2);const o=this.kernelSize[0],a=this.kernelSize[1],l=this.strides[0],u=this.strides[1];return t[n]=this.filters,t[r]=Yn(t[r],l,o,this.padding),t[i]=Yn(t[i],u,a,this.padding),t}getConfig(){const e=super.getConfig();return delete e.dilationRate,e}}zm.className="Conv2DTranspose";le(zm);class Wm extends Lo{constructor(e){if(super(e),this.inputSpec=[new Jn({ndim:5})],this.padding!=="same"&&this.padding!=="valid")throw new q(`Conv3DTranspose currently supports only padding modes 'same' and 'valid', but received padding mode ${this.padding}`)}build(e){if(e=Nn(e),e.length!==5)throw new q("Input should have rank 5; Received input shape: "+JSON.stringify(e));const t=this.dataFormat==="channelsFirst"?1:e.length-1;if(e[t]==null)throw new q("The channel dimension of the inputs should be defined. Found `None`.");const n=e[t],r=this.kernelSize.concat([this.filters,n]);this.kernel=this.addWeight("kernel",r,"float32",this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight("bias",[this.filters],"float32",this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint)),this.inputSpec=[new Jn({ndim:5,axes:{[t]:n}})],this.built=!0}call(e,t){return Q(()=>{let n=zt(e);if(n.shape.length!==5)throw new q(`Conv3DTranspose.call() expects input tensor to be rank-4, but received a tensor of rank-${n.shape.length}`);const r=n.shape,i=r[0];let o,a,l;this.dataFormat==="channelsFirst"?(l=2,o=3,a=4):(l=1,o=2,a=3);const u=r[l],c=r[o],h=r[a],d=this.kernelSize[0],w=this.kernelSize[1],k=this.kernelSize[2],A=this.strides[0],m=this.strides[1],S=this.strides[2],b=Yn(u,A,d,this.padding),f=Yn(c,m,w,this.padding),v=Yn(h,S,k,this.padding),_=[i,b,f,v,this.filters];this.dataFormat!=="channelsLast"&&(n=He(n,[0,2,3,4,1]));let E=Vw(n,this.kernel.read(),_,this.strides,this.padding);return this.dataFormat!=="channelsLast"&&(E=He(E,[0,4,1,2,3])),this.bias!==null&&(E=Po(E,this.bias.read(),this.dataFormat)),this.activation!==null&&(E=this.activation.apply(E)),E})}computeOutputShape(e){e=Nn(e);const t=e.slice();let n,r,i,o;this.dataFormat==="channelsFirst"?(n=1,r=2,i=3,o=4):(n=4,r=1,i=2,o=3);const a=this.kernelSize[0],l=this.kernelSize[1],u=this.kernelSize[2],c=this.strides[0],h=this.strides[1],d=this.strides[2];return t[n]=this.filters,t[r]=Yn(t[r],c,a,this.padding),t[i]=Yn(t[i],h,l,this.padding),t[o]=Yn(t[o],d,u,this.padding),t}getConfig(){const e=super.getConfig();return delete e.dilationRate,e}}Wm.className="Conv3DTranspose";le(Wm);class Gm extends Ii{constructor(e,t){if(super(e,t),this.DEFAULT_DEPTHWISE_INITIALIZER="glorotUniform",this.DEFAULT_POINTWISE_INITIALIZER="glorotUniform",this.depthwiseKernel=null,this.pointwiseKernel=null,t.filters==null)throw new q("The `filters` configuration field is required by SeparableConv, but is unspecified.");if(t.kernelInitializer!=null||t.kernelRegularizer!=null||t.kernelConstraint!=null)throw new q("Fields kernelInitializer, kernelRegularizer and kernelConstraint are invalid for SeparableConv2D. Use depthwiseInitializer, depthwiseRegularizer, depthwiseConstraint, pointwiseInitializer, pointwiseRegularizer and pointwiseConstraint instead.");if(t.padding!=null&&t.padding!=="same"&&t.padding!=="valid")throw new q(`SeparableConv${this.rank}D supports only padding modes: 'same' and 'valid', but received ${JSON.stringify(t.padding)}`);this.depthMultiplier=t.depthMultiplier==null?1:t.depthMultiplier,this.depthwiseInitializer=_o(t.depthwiseInitializer||this.DEFAULT_DEPTHWISE_INITIALIZER),this.depthwiseRegularizer=So(t.depthwiseRegularizer),this.depthwiseConstraint=Qa(t.depthwiseConstraint),this.pointwiseInitializer=_o(t.depthwiseInitializer||this.DEFAULT_POINTWISE_INITIALIZER),this.pointwiseRegularizer=So(t.pointwiseRegularizer),this.pointwiseConstraint=Qa(t.pointwiseConstraint)}build(e){if(e=Nn(e),e.length<this.rank+2)throw new q(`Inputs to SeparableConv${this.rank}D should have rank ${this.rank+2}, but received input shape: ${JSON.stringify(e)}`);const t=this.dataFormat==="channelsFirst"?1:e.length-1;if(e[t]==null||e[t]<0)throw new q(`The channel dimension of the inputs should be defined, but found ${JSON.stringify(e[t])}`);const n=e[t],r=this.kernelSize.concat([n,this.depthMultiplier]),i=[];for(let a=0;a<this.rank;++a)i.push(1);i.push(n*this.depthMultiplier,this.filters);const o=!0;this.depthwiseKernel=this.addWeight("depthwise_kernel",r,"float32",this.depthwiseInitializer,this.depthwiseRegularizer,o,this.depthwiseConstraint),this.pointwiseKernel=this.addWeight("pointwise_kernel",i,"float32",this.pointwiseInitializer,this.pointwiseRegularizer,o,this.pointwiseConstraint),this.useBias?this.bias=this.addWeight("bias",[this.filters],"float32",this.biasInitializer,this.biasRegularizer,o,this.biasConstraint):this.bias=null,this.inputSpec=[new Jn({ndim:this.rank+2,axes:{[t]:n}})],this.built=!0}call(e,t){return Q(()=>{e=zt(e);let n;if(this.rank===1)throw new Se("1D separable convolution is not implemented yet.");return this.rank===2&&(this.dataFormat==="channelsFirst"&&(e=He(e,[0,2,3,1])),n=C2(e,this.depthwiseKernel.read(),this.pointwiseKernel.read(),this.strides,this.padding,this.dilationRate,"NHWC")),this.useBias&&(n=Po(n,this.bias.read(),this.dataFormat)),this.activation!=null&&(n=this.activation.apply(n)),this.dataFormat==="channelsFirst"&&(n=He(n,[0,3,1,2])),n})}getConfig(){const e=super.getConfig();return delete e.rank,delete e.kernelInitializer,delete e.kernelRegularizer,delete e.kernelConstraint,e.depthwiseInitializer=Ja(this.depthwiseInitializer),e.pointwiseInitializer=Ja(this.pointwiseInitializer),e.depthwiseRegularizer=vo(this.depthwiseRegularizer),e.pointwiseRegularizer=vo(this.pointwiseRegularizer),e.depthwiseConstraint=Za(this.depthwiseConstraint),e.pointwiseConstraint=Za(this.pointwiseConstraint),e}}Gm.className="SeparableConv";class Vm extends Gm{constructor(e){super(2,e)}}Vm.className="SeparableConv2D";le(Vm);class Pl extends Ii{constructor(e){super(1,e),Pl.verifyArgs(e),this.inputSpec=[{ndim:3}]}getConfig(){const e=super.getConfig();return delete e.rank,delete e.dataFormat,e}static verifyArgs(e){if(typeof e.kernelSize!="number"&&!Rc(e.kernelSize,"number",1,1))throw new q(`Conv1D expects config.kernelSize to be number or number[] with length 1, but received ${JSON.stringify(e.kernelSize)}.`)}}Pl.className="Conv1D";le(Pl);class qm extends $n{constructor(e){super(e),typeof e.cropping=="number"?this.cropping=[[e.cropping,e.cropping],[e.cropping,e.cropping]]:typeof e.cropping[0]=="number"?this.cropping=[[e.cropping[0],e.cropping[0]],[e.cropping[1],e.cropping[1]]]:this.cropping=e.cropping,this.dataFormat=e.dataFormat===void 0?"channelsLast":e.dataFormat,this.inputSpec=[{ndim:4}]}computeOutputShape(e){return this.dataFormat==="channelsFirst"?[e[0],e[1],e[2]-this.cropping[0][0]-this.cropping[0][1],e[3]-this.cropping[1][0]-this.cropping[1][1]]:[e[0],e[1]-this.cropping[0][0]-this.cropping[0][1],e[2]-this.cropping[1][0]-this.cropping[1][1],e[3]]}call(e,t){return Q(()=>{if(e=zt(e),this.dataFormat==="channelsLast"){const n=Qo(e,this.cropping[0][0],e.shape[1]-this.cropping[0][0]-this.cropping[0][1],2);return Qo(n,this.cropping[1][0],e.shape[2]-this.cropping[1][1]-this.cropping[1][0],3)}else{const n=Qo(e,this.cropping[0][0],e.shape[2]-this.cropping[0][0]-this.cropping[0][1],3);return Qo(n,this.cropping[1][0],e.shape[3]-this.cropping[1][1]-this.cropping[1][0],4)}})}getConfig(){const e={cropping:this.cropping,dataFormat:this.dataFormat},t=super.getConfig();return Object.assign(e,t),e}}qm.className="Cropping2D";le(qm);class Kc extends $n{constructor(e){super(e),this.DEFAULT_SIZE=[2,2],this.inputSpec=[{ndim:4}],this.size=e.size==null?this.DEFAULT_SIZE:e.size,this.dataFormat=e.dataFormat==null?"channelsLast":e.dataFormat,Ze(this.dataFormat),this.interpolation=e.interpolation==null?"nearest":e.interpolation,Mv(this.interpolation)}computeOutputShape(e){if(this.dataFormat==="channelsFirst"){const t=e[2]==null?null:this.size[0]*e[2],n=e[3]==null?null:this.size[1]*e[3];return[e[0],e[1],t,n]}else{const t=e[1]==null?null:this.size[0]*e[1],n=e[2]==null?null:this.size[1]*e[2];return[e[0],t,n,e[3]]}}call(e,t){return Q(()=>{let n=zt(e);const r=n.shape;if(this.dataFormat==="channelsFirst"){n=He(n,[0,2,3,1]);const i=this.size[0]*r[2],o=this.size[1]*r[3],a=this.interpolation==="nearest"?Yo.resizeNearestNeighbor(n,[i,o]):Yo.resizeBilinear(n,[i,o]);return He(a,[0,3,1,2])}else{const i=this.size[0]*r[1],o=this.size[1]*r[2];return this.interpolation==="nearest"?Yo.resizeNearestNeighbor(n,[i,o]):Yo.resizeBilinear(n,[i,o])}})}getConfig(){const e={size:this.size,dataFormat:this.dataFormat,interpolation:this.interpolation},t=super.getConfig();return Object.assign(e,t),e}}Kc.className="UpSampling2D";le(Kc);/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function Rl(s,e,t,n,r,i){return Q(()=>{Ze(r),em(i),hn(n),t==null&&(t=[1,1]),n==null&&(n="valid"),r==null&&(r=ki()),i==null&&(i="max"),s=Fm(s,r);let o;const a=n==="same"?"same":"valid";return i==="max"?o=Ux(s,e,t,a):o=Sw(s,e,t,a),r==="channelsFirst"&&(o=He(o,[0,3,1,2])),o})}function Hm(s,e,t,n,r,i){return Q(()=>{Ze(r),em(i),hn(n),t==null&&(t=[1,1,1]),n==null&&(n="valid"),r==null&&(r=ki()),i==null&&(i="max"),s=Um(s,r);let o;const a=n==="same"?"same":"valid";return i==="max"?o=Wx(s,e,t,a):o=Iw(s,e,t,a),r==="channelsFirst"&&(o=He(o,[0,4,1,2,3])),o})}class jm extends $n{constructor(e){if(e.poolSize==null&&(e.poolSize=2),super(e),typeof e.poolSize=="number")this.poolSize=[e.poolSize];else if(Array.isArray(e.poolSize)&&e.poolSize.length===1&&typeof e.poolSize[0]=="number")this.poolSize=e.poolSize;else throw new q(`poolSize for 1D convolutional layer must be a number or an Array of a single number, but received ${JSON.stringify(e.poolSize)}`);if(ys(this.poolSize,"poolSize"),e.strides==null)this.strides=this.poolSize;else if(typeof e.strides=="number")this.strides=[e.strides];else if(Array.isArray(e.strides)&&e.strides.length===1&&typeof e.strides[0]=="number")this.strides=e.strides;else throw new q(`strides for 1D convolutional layer must be a number or an Array of a single number, but received ${JSON.stringify(e.strides)}`);ys(this.strides,"strides"),this.padding=e.padding==null?"valid":e.padding,hn(this.padding),this.inputSpec=[new Jn({ndim:3})]}computeOutputShape(e){e=Nn(e);const t=br(e[1],this.poolSize[0],this.padding,this.strides[0]);return[e[0],t,e[2]]}call(e,t){return Q(()=>{this.invokeCallHook(e,t),e=Lc(zt(e),2);const n=this.poolingFunction(zt(e),[this.poolSize[0],1],[this.strides[0],1],this.padding,"channelsLast");return Ml(n,[2])})}getConfig(){const e={poolSize:this.poolSize,padding:this.padding,strides:this.strides},t=super.getConfig();return Object.assign(e,t),e}}class Km extends jm{constructor(e){super(e)}poolingFunction(e,t,n,r,i){return Ze(i),hn(r),Rl(e,t,n,r,i,"max")}}Km.className="MaxPooling1D";le(Km);class Xm extends jm{constructor(e){super(e)}poolingFunction(e,t,n,r,i){return Ze(i),hn(r),Rl(e,t,n,r,i,"avg")}}Xm.className="AveragePooling1D";le(Xm);class Ym extends $n{constructor(e){if(e.poolSize==null&&(e.poolSize=[2,2]),super(e),this.poolSize=Array.isArray(e.poolSize)?e.poolSize:[e.poolSize,e.poolSize],e.strides==null)this.strides=this.poolSize;else if(Array.isArray(e.strides)){if(e.strides.length!==2)throw new q(`If the strides property of a 2D pooling layer is an Array, it is expected to have a length of 2, but received length ${e.strides.length}.`);this.strides=e.strides}else this.strides=[e.strides,e.strides];ys(this.poolSize,"poolSize"),ys(this.strides,"strides"),this.padding=e.padding==null?"valid":e.padding,this.dataFormat=e.dataFormat==null?"channelsLast":e.dataFormat,Ze(this.dataFormat),hn(this.padding),this.inputSpec=[new Jn({ndim:4})]}computeOutputShape(e){e=Nn(e);let t=this.dataFormat==="channelsFirst"?e[2]:e[1],n=this.dataFormat==="channelsFirst"?e[3]:e[2];return t=br(t,this.poolSize[0],this.padding,this.strides[0]),n=br(n,this.poolSize[1],this.padding,this.strides[1]),this.dataFormat==="channelsFirst"?[e[0],e[1],t,n]:[e[0],t,n,e[3]]}call(e,t){return Q(()=>(this.invokeCallHook(e,t),this.poolingFunction(zt(e),this.poolSize,this.strides,this.padding,this.dataFormat)))}getConfig(){const e={poolSize:this.poolSize,padding:this.padding,strides:this.strides,dataFormat:this.dataFormat},t=super.getConfig();return Object.assign(e,t),e}}class Xc extends Ym{constructor(e){super(e)}poolingFunction(e,t,n,r,i){return Ze(i),hn(r),Rl(e,t,n,r,i,"max")}}Xc.className="MaxPooling2D";le(Xc);class Zm extends Ym{constructor(e){super(e)}poolingFunction(e,t,n,r,i){return Ze(i),hn(r),Rl(e,t,n,r,i,"avg")}}Zm.className="AveragePooling2D";le(Zm);class Qm extends $n{constructor(e){if(e.poolSize==null&&(e.poolSize=[2,2,2]),super(e),this.poolSize=Array.isArray(e.poolSize)?e.poolSize:[e.poolSize,e.poolSize,e.poolSize],e.strides==null)this.strides=this.poolSize;else if(Array.isArray(e.strides)){if(e.strides.length!==3)throw new q(`If the strides property of a 3D pooling layer is an Array, it is expected to have a length of 3, but received length ${e.strides.length}.`);this.strides=e.strides}else this.strides=[e.strides,e.strides,e.strides];ys(this.poolSize,"poolSize"),ys(this.strides,"strides"),this.padding=e.padding==null?"valid":e.padding,this.dataFormat=e.dataFormat==null?"channelsLast":e.dataFormat,Ze(this.dataFormat),hn(this.padding),this.inputSpec=[new Jn({ndim:5})]}computeOutputShape(e){e=Nn(e);let t=this.dataFormat==="channelsFirst"?e[2]:e[1],n=this.dataFormat==="channelsFirst"?e[3]:e[2],r=this.dataFormat==="channelsFirst"?e[4]:e[3];return t=br(t,this.poolSize[0],this.padding,this.strides[0]),n=br(n,this.poolSize[1],this.padding,this.strides[1]),r=br(r,this.poolSize[2],this.padding,this.strides[2]),this.dataFormat==="channelsFirst"?[e[0],e[1],t,n,r]:[e[0],t,n,r,e[4]]}call(e,t){return Q(()=>(this.invokeCallHook(e,t),this.poolingFunction(zt(e),this.poolSize,this.strides,this.padding,this.dataFormat)))}getConfig(){const e={poolSize:this.poolSize,padding:this.padding,strides:this.strides,dataFormat:this.dataFormat},t=super.getConfig();return Object.assign(e,t),e}}class Jm extends Qm{constructor(e){super(e)}poolingFunction(e,t,n,r,i){return Ze(i),hn(r),Hm(e,t,n,r,i,"max")}}Jm.className="MaxPooling3D";le(Jm);class eg extends Qm{constructor(e){super(e)}poolingFunction(e,t,n,r,i){return Ze(i),hn(r),Hm(e,t,n,r,i,"avg")}}eg.className="AveragePooling3D";le(eg);class tg extends $n{constructor(e){super(e),this.inputSpec=[new Jn({ndim:3})]}computeOutputShape(e){return[e[0],e[2]]}call(e,t){throw new Se}}class ng extends tg{constructor(e){super(e||{})}call(e,t){return Q(()=>{const n=zt(e);return ut(n,1)})}}ng.className="GlobalAveragePooling1D";le(ng);class sg extends tg{constructor(e){super(e||{})}call(e,t){return Q(()=>{const n=zt(e);return Ds(n,1)})}}sg.className="GlobalMaxPooling1D";le(sg);class rg extends $n{constructor(e){super(e),this.dataFormat=e.dataFormat==null?"channelsLast":e.dataFormat,Ze(this.dataFormat),this.inputSpec=[new Jn({ndim:4})]}computeOutputShape(e){return e=e,this.dataFormat==="channelsLast"?[e[0],e[3]]:[e[0],e[1]]}call(e,t){throw new Se}getConfig(){const e={dataFormat:this.dataFormat},t=super.getConfig();return Object.assign(e,t),e}}class ig extends rg{call(e,t){return Q(()=>{const n=zt(e);return this.dataFormat==="channelsLast"?ut(n,[1,2]):ut(n,[2,3])})}}ig.className="GlobalAveragePooling2D";le(ig);class og extends rg{call(e,t){return Q(()=>{const n=zt(e);return this.dataFormat==="channelsLast"?Ds(n,[1,2]):Ds(n,[2,3])})}}og.className="GlobalMaxPooling2D";le(og);/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function nl(s,e){return Q(()=>{s.dtype!=="float32"&&(s=De(s,"float32"));const t=Ae(Mo(s),e,!0),n=Dl(t.shape,Ye()),r=Cn(xi(t,n));return _e(s,r)})}function Ll(s,e){return Q(()=>ut(Mo(ke(e,s)),-1))}function Yc(s,e){return Q(()=>ut(Tt(ke(e,s)),-1))}function Zc(s,e){return Q(()=>{const t=ke(s,e),n=An(Tt(s),Ye(),Number.MAX_VALUE),r=Tt(_e(t,n));return ne(100,ut(r,-1))})}function c3(s,e){return Q(()=>{const t=An(e,Ye(),Number.MAX_VALUE),n=Ir(he(1,t)),r=An(s,Ye(),Number.MAX_VALUE),i=Ir(he(1,r));return ut(Mo(ke(n,i)),-1)})}function h3(s,e){return Q(()=>{const t=xi(0,ke(1,ne(s,e)));return ut(Mo(t),-1)})}function f3(s,e){return Q(()=>{const t=xi(0,ke(1,ne(s,e)));return ut(t,-1)})}function d3(s,e){return Q(()=>{const t=Ae(ne(s,e),-1),n=Ds(ne(ke(1,s),e),-1);return xi(0,he(1,ke(n,t)))})}function p3(s,e){return Q(()=>{const t=Math.log(2),n=ke(e,s),r=ke(he(n,Ac(ne(-2,n))),t);return ut(r,-1)})}function ko(s,e,t=!1){return Q(()=>{if(t)e=Wp(e);else{const n=Ae(e,e.shape.length-1,!0);e=_e(e,n)}return e=An(e,Ye(),1-Ye()),wi(Ae(ne(De(s,"float32"),Ir(e)),e.shape.length-1))})}function sl(s,e,t=!1){return Q(()=>{const n=De(yx(Bv(s)),"int32");e=An(e,Ye(),1-Ye());const r=e.shape,i=ae(jx(n,r[r.length-1]),r);return ko(i,e,t)})}function m3(s,e){if(!cn(s.shape,e.shape))throw new q(`logits and labels must have the same shape, but got shapes ${JSON.stringify(s.shape)} and ${JSON.stringify(e.shape)}`);return Q(()=>{const t=Do(e),n=wi(Tt(e));return he(ke(t,ne(e,s)),$x(Fu(n)))})}function Bl(s,e){return Q(()=>{let t;return t=An(e,Ye(),1-Ye()),t=Ir(_e(t,ke(1,t))),ut(m3(s,t),-1)})}function g3(s,e){return Q(()=>{const t=An(s,Ye(),1),n=An(e,Ye(),1);return Ae(ne(s,Ir(_e(t,n))),-1)})}function y3(s,e){return Q(()=>{const t=Ir(he(Ye(),e));return ut(ke(e,ne(s,t)),-1)})}function ag(s,e){return Q(()=>{const t=nl(s,-1),n=nl(e,-1),r=ne(t,n);return wi(Ae(r,-1))})}const rl={meanSquaredError:Ll,meanAbsoluteError:Yc,meanAbsolutePercentageError:Zc,meanSquaredLogarithmicError:c3,squaredHinge:h3,hinge:f3,categoricalHinge:d3,logcosh:p3,categoricalCrossentropy:ko,sparseCategoricalCrossentropy:sl,binaryCrossentropy:Bl,kullbackLeiblerDivergence:g3,poisson:y3,cosineProximity:ag};function lu(s){if(typeof s=="string"){if(s in rl)return rl[s];let e=`Unknown loss ${s}`;throw s.toLowerCase().includes("softmaxcrossentropy")&&(e=`Unknown loss ${s}. Use "categoricalCrossentropy" as the string name for tf.losses.softmaxCrossEntropy`),new q(e)}else return s}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */class Nr extends $n{constructor(e){super(e||{}),this.supportsMasking=!0}mergeFunction(e){throw new Se}computeElementwiseOpOutputShape(e,t){if(e==null||t==null)return null;if(e.length<t.length)return this.computeElementwiseOpOutputShape(t,e);if(t.length===0)return e;const n=e.slice(0,e.length-t.length);for(let r=0;r<t.length;++r){const i=e[e.length-t.length+r],o=t[r];if(i==null||o==null||i<0||o<0)n.push(null);else if(i===1)n.push(o);else if(o===1)n.push(i);else{if(i!==o)throw new q("Operands could not be broadcast together with shapes "+JSON.stringify(e)+" "+JSON.stringify(t));n.push(i)}}return n}build(e){if(Array.isArray(e)&&!Array.isArray(e[0])&&(e=[Nn(e)]),e=e,e.length<2)throw new q(`A merge layer should be called on an Array of at least 2 inputs. Got ${e.length} input(s).`);let t=[];for(const i of e)i!=null&&i[0]!==null&&t.push(i[0]);if(t=gr(t),t.length>1)throw new q(`Can not merge tensors with different batch sizes. Got tensors with shapes: ${JSON.stringify(e)}.`);let n=e[0]==null?null:e[0].slice(1);for(let i=1;i<e.length;++i){const o=e[i]==null?null:e[i].slice(1);n=this.computeElementwiseOpOutputShape(n,o)}const r=e.map(i=>i.length);e.indexOf(null)===-1&&gr(r).length===1?this.reshapeRequired=!1:this.reshapeRequired=!0}call(e,t){return Q(()=>{if(e=e,this.reshapeRequired){const n=[],r=e.map(i=>i.rank);if(r.indexOf(null)===-1){const i=rm(r);for(let o of e){const a=o.rank;for(let l=0;l<i-a;++l)o=Lc(o,1);n.push(o)}return this.mergeFunction(n)}else{let i=!1;for(const l of e){const u=l.rank;if(u==null){const c=l.shape,h=c[0],d=c.slice(1).concat([h]);let w=ae(l,[h].concat(no(c.slice(1))));w=He(w,[1,0]),w=ae(w,d),n.push(w),i=!0}else if(u>1){const c=Ya(1,u).concat([0]);n.push(He(l,c)),i=!0}else n.push(l)}let o=this.mergeFunction(n);const a=o.rank;if(i){if(a==null){const l=o.shape,u=l.length,c=l[u-1],h=[c].concat(l.slice(0,l.length-1));o=ae(He(ae(o,[-1,c]),[1,0]),h)}else if(a>1){const l=[a-1].concat(Ya(0,a-1));o=He(o,l)}}return o}}else return this.mergeFunction(e)})}computeOutputShape(e){e=e;let t;e[0]==null?t=null:t=e[0].slice(1);for(let r=1;r<e.length;++r){const i=e[r]==null?null:e[r].slice(1);t=this.computeElementwiseOpOutputShape(t,i)}let n=[];for(const r of e)r!=null&&r[0]!==null&&n.push(r[0]);return n=gr(n),n.length===1?t=n.concat(t):t=[null].concat(t),t}computeMask(e,t){return Q(()=>{if(t==null)return null;if(!Array.isArray(t))throw new q("`mask` should be an Array");if(!Array.isArray(e))throw new q("`inputs` should be an Array");if(t.length!==e.length)throw new q(`The Array 'inputs' and 'mask' are expected to have the same length, but have different lengths (${e.length} vs ${t.length})`);if(t.every(r=>r==null))return null;t=t.map(r=>r==null?r:jn(r,0));let n=t[0];for(let r=1;r<t.length-1;++r)n=Ol(n,t[r]);return n})}}class lg extends Nr{constructor(e){super(e)}mergeFunction(e){return Q(()=>{let t=e[0].clone();for(let n=1;n<e.length;++n)t=he(t,e[n]);return t})}}lg.className="Add";le(lg);class ug extends Nr{constructor(e){super(e)}mergeFunction(e){return Q(()=>{let t=e[0].clone();for(let n=1;n<e.length;++n)t=ne(t,e[n]);return t})}}ug.className="Multiply";le(ug);class cg extends Nr{constructor(e){super(e)}mergeFunction(e){return Q(()=>{let t=e[0].clone();for(let n=1;n<e.length;++n)t=he(t,e[n]);return ne(1/e.length,t)})}}cg.className="Average";le(cg);class hg extends Nr{constructor(e){super(e)}mergeFunction(e){return Q(()=>{let t=e[0];for(let n=1;n<e.length;++n)t=xi(t,e[n]);return t})}}hg.className="Maximum";le(hg);class fg extends Nr{constructor(e){super(e)}mergeFunction(e){return Q(()=>{let t=e[0];for(let n=1;n<e.length;++n)t=Ha(t,e[n]);return t})}}fg.className="Minimum";le(fg);class Qc extends Nr{constructor(e){super(e),this.DEFAULT_AXIS=-1,e==null&&(e={}),this.axis=e.axis==null?this.DEFAULT_AXIS:e.axis,this.supportsMasking=!0,this.reshapeRequired=!1}build(e){if(!(Array.isArray(e)&&Array.isArray(e[0]))||e.length===1)throw new q("A `Concatenate` layer should be called on a list of at least 2 inputs");e=e;let t=!0;for(const r of e)if(r!=null){t=!1;break}if(t)return;const n=[];for(let r=0;r<e.length;++r){const i=e[r].slice();i.splice(this.axis,1);let o=!1;for(const a of n)if(cn(a,i)){o=!0;break}o||n.push(i)}if(n.length>1)throw new q("A `Concatenate` layer requires inputs with matching shapes except for the concat axis. Got input shapes: "+JSON.stringify(e))}mergeFunction(e){return Q(()=>Fv(e,this.axis))}computeOutputShape(e){if(!(Array.isArray(e)&&Array.isArray(e[0])))throw new q("A `Concatenate` layer should be called on a list of inputs.");const t=e,n=t[0].slice(),r=this.axis<0?n.length+this.axis:this.axis;for(const i of t.slice(1)){if(n[r]==null||i[r]==null){n[r]=null;break}n[r]+=i[r]}return n}computeMask(e,t){if(t==null)return null;if(!Array.isArray(t))throw new q("`mask` should be an array for Concatenate");if(!Array.isArray(e))throw new q("`inputs` should be an array for Concatenate");if(t.length!==e.length)throw new q(`Mismatch in the length of mask (${t.length}) and the legnth of inputs (${e.length})`);return Q(()=>{let n=!0;if(t.forEach(o=>{if(o!=null){n=!1;return}}),n)return null;const r=[];for(let o=0;o<e.length;++o)t[o]==null?r.push(De(Lp(e[o]),"bool")):t[o].rank<e[o].rank?r.push(jn(t[o],-1)):r.push(t[o]);const i=pr(r,this.axis);return pw(i,-1,!1)})}getConfig(){const e={axis:this.axis},t=super.getConfig();return Object.assign(e,t),e}}Qc.className="Concatenate";le(Qc);function Di(s,e){for(;s<0;)s+=e;return s}function b3(s,e,t){if(s.shape.length>3||e.shape.length>3)throw new Se("batchDot is not implemented for tensors of 4D or higher rank yet");if(R(s.shape.length>=2,()=>`batchDot requires the rank of x to be >= 2, but got ${s.shape.length}`),R(s.shape.length>=2,()=>`batchDot requires the rank of y to be >= 2, but got ${e.shape.length}`),typeof t=="number"&&(t=[t,t]),s.dtype==="complex64"||e.dtype==="complex64")throw new Se("batchDot is not implemented for complex64-type Tensors yet.");const n=s.shape.length,r=e.shape.length;t==null&&(t=[n-1,r-2]);const i=t;return Q(()=>{let o;if(n>r){o=n-r;const l=[];for(let u=0;u<o;++u)l.push(1);e=ae(e,e.shape.concat(l))}else if(r>n){o=r-n;const l=[];for(let u=0;u<o;++u)l.push(1);s=ae(s,s.shape.concat(l))}else o=0;let a;if(s.shape.length===2&&e.shape.length===2)i[0]===i[1]?a=Ae(ne(s,e),i[0]):a=Ae(ne(He(s,[1,0]),e),i[1]);else{const l=i[0]!==s.shape.length-1,u=i[1]===e.shape.length-1;a=Pn(s,e,l,u)}if(o>0){let l;n>r?l=n+r-3:l=n-1;const u=[];for(let c=l;c<l+o;++c)u.push(c);a=Ml(a,u)}return a.shape.length===1&&(a=jn(a,1)),a})}class dg extends Nr{constructor(e){super(e),this.axes=e.axes,this.normalize=e.normalize==null?!1:e.normalize,this.supportsMasking=!0,this.reshapeRequired=!1}build(e){R(Array.isArray(e)&&e.length===2&&Array.isArray(e[0])&&Array.isArray(e[1]),()=>"A `Dot` layer should be called on a list of exactly 2 inputs.");const t=e[0],n=e[1];if(t.length>3||n.length>3)throw new Se("Dot layer does not support tensors of 4D or higher rank yet.");const r=this.interpretAxes(t,n);if(t[r[0]]!==n[r[1]])throw new q(`Dimension incompatibility: ${t[r[0]]} !== ${n[r[1]]}`)}mergeFunction(e){if(e.length!==2)throw new q(`A \`Dot\` layer must be called on exactly 2 inputs, but received ${e.length} input(s).`);let t=e[0],n=e[1],r;return Array.isArray(this.axes)?r=this.axes.map((i,o)=>Di(i,e[o].shape.length)):r=[Di(this.axes,t.shape.length),Di(this.axes,n.shape.length)],this.normalize&&(t=nl(t,r[0]),n=nl(n,r[1])),b3(t,n,r)}interpretAxes(e,t){let n;return Array.isArray(this.axes)?n=this.axes:n=[Di(this.axes,e.length),Di(this.axes,t.length)],n}computeOutputShape(e){R(Array.isArray(e)&&e.length===2&&Array.isArray(e[0])&&Array.isArray(e[1]),()=>"A `Dot` layer should be called on a list of exactly 2 inputs.");const t=e[0].slice(),n=e[1].slice();if(t.length>3||n.length>3)throw new Se("Dot layer does not support tensors of 4D or higher rank yet.");const r=this.interpretAxes(t,n);t.splice(r[0],1),n.splice(r[1],1),n.splice(0,1);const i=t.concat(n);return i.length===1&&i.push(1),i}computeMask(e,t){return null}getConfig(){const e={axes:this.axes,normalize:this.normalize},t=super.getConfig();return Object.assign(e,t),e}}dg.className="Dot";le(dg);/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */async function Fs(s){if(s==null)return;const e=[],t=[],n=[];for(const r in s){const i=s[r];if(typeof i!="number"){const o=i;e.push(o.data()),t.push(r),n.push(o)}}if(e.length>0){const r=await Promise.all(e);for(let i=0;i<r.length;++i)s[t[i]]=r[i][0];Pe(n)}}function pg(s){if(s!=null)for(const e in s){const t=s[e];typeof t!="number"&&t.dispose()}}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */var vf;(function(s){s[s.SILENT=0]="SILENT",s[s.VERBOSE=1]="VERBOSE"})(vf||(vf={}));const w3=125;class Io{constructor(){this.validationData=null}setParams(e){this.params=e}async onEpochBegin(e,t){}async onEpochEnd(e,t){}async onBatchBegin(e,t){}async onBatchEnd(e,t){}async onTrainBegin(e){}async onTrainEnd(e){}setModel(e){}}class x3{constructor(e,t=10){e==null&&(e=[]),this.callbacks=e,this.queueLength=t}append(e){this.callbacks.push(e)}setParams(e){for(const t of this.callbacks)t.setParams(e)}setModel(e){for(const t of this.callbacks)t.setModel(e)}async onEpochBegin(e,t){t==null&&(t={});for(const n of this.callbacks)await n.onEpochBegin(e,t)}async onEpochEnd(e,t){t==null&&(t={});for(const n of this.callbacks)await n.onEpochEnd(e,t)}async onBatchBegin(e,t){t==null&&(t={});for(const n of this.callbacks)await n.onBatchBegin(e,t)}async onBatchEnd(e,t){t==null&&(t={});for(const n of this.callbacks)await n.onBatchEnd(e,t)}async onTrainBegin(e){e==null&&(e={});for(const t of this.callbacks)await t.onTrainBegin(e)}async onTrainEnd(e){e==null&&(e={});for(const t of this.callbacks)await t.onTrainEnd(e)}}class _3 extends Io{constructor(){super()}async onEpochBegin(e){this.seen=0,this.totals={}}async onBatchEnd(e,t){t==null&&(t={});const n=t.size==null?0:t.size;this.seen+=n;for(const r in t){const i=t[r];if(typeof i=="number")this.totals.hasOwnProperty(r)||(this.totals[r]=0),this.totals[r]=this.totals[r]+i*n;else{let o;r in this.totals?o=this.totals[r]:this.totals[r]=0;const a=Q(()=>he(this.totals[r],ne(i,n)));this.totals[r]=a,o?.dispose()}}}async onEpochEnd(e,t){if(t!=null)for(const n of this.params.metrics)this.totals[n]!=null&&(typeof this.totals[n]=="number"?t[n]=this.totals[n]/this.seen:Q(()=>{const r=ne(_e(1,this.seen),this.totals[n]);t[n]=r,this.totals[n].dispose(),oi(t[n])}))}}class v3 extends Io{async onTrainBegin(e){this.epoch=[],this.history={}}async onEpochEnd(e,t){t==null&&(t={}),this.epoch.push(e);for(const n in t)this.history[n]==null&&(this.history[n]=[]),this.history[n].push(t[n])}async syncData(){const e=[],t=[],n=[];for(const i in this.history){const o=this.history[i];for(let a=0;a<o.length;++a)if(typeof o[a]!="number"){const l=o[a];e.push(l.data()),t.push(i),n.push(a)}}const r=await Promise.all(e);for(let i=0;i<r.length;++i)this.history[t[i]][n[i]].dispose(),this.history[t[i]][n[i]]=r[i][0]}}class S3 extends Io{constructor(e,t){if(super(),this.currentEpoch=0,this.nowFunc=e.nowFunc,this.nextFrameFunc=e.nextFrameFunc||ev,this.yieldEvery=t||"auto",this.yieldEvery==="auto"&&(this.yieldEvery=w3),this.yieldEvery==="never"&&e.onYield!=null)throw new Error("yieldEvery is `never` but you provided an `onYield` callback. Either change `yieldEvery` or remove the callback");Iu(this.yieldEvery)&&(this.maybeWait=Dv(this.maybeWait.bind(this),this.yieldEvery,this.nowFunc)),this.trainBegin=e.onTrainBegin,this.trainEnd=e.onTrainEnd,this.epochBegin=e.onEpochBegin,this.epochEnd=e.onEpochEnd,this.batchBegin=e.onBatchBegin,this.batchEnd=e.onBatchEnd,this.yield=e.onYield}async maybeWait(e,t,n){const r=[];this.yield!=null&&(await Fs(n),r.push(this.yield(e,t,n))),r.push(this.nextFrameFunc()),await Promise.all(r)}async onEpochBegin(e,t){this.currentEpoch=e,this.epochBegin!=null&&(await Fs(t),await this.epochBegin(e,t))}async onEpochEnd(e,t){const n=[];this.epochEnd!=null&&(await Fs(t),n.push(this.epochEnd(e,t))),this.yieldEvery==="epoch"&&n.push(this.nextFrameFunc()),await Promise.all(n)}async onBatchBegin(e,t){this.batchBegin!=null&&(await Fs(t),await this.batchBegin(e,t))}async onBatchEnd(e,t){const n=[];this.batchEnd!=null&&(await Fs(t),n.push(this.batchEnd(e,t))),this.yieldEvery==="batch"?n.push(this.nextFrameFunc()):Iu(this.yieldEvery)&&n.push(this.maybeWait(this.currentEpoch,e,t)),await Promise.all(n)}async onTrainBegin(e){this.trainBegin!=null&&(await Fs(e),await this.trainBegin(e))}async onTrainEnd(e){this.trainEnd!=null&&(await Fs(e),await this.trainEnd(e))}}function mg(s,e){return s==null&&(s={}),s instanceof Io?[s]:Array.isArray(s)&&s[0]instanceof Io?s:Ne(s).map(n=>new S3(n,e))}class rn{constructor(){}static registerCallbackConstructor(e,t){R(e>=0&&Number.isInteger(e),()=>`Verbosity level is expected to be an integer >= 0, but got ${e}`),rn.checkForDuplicate(t),rn.constructors[e]==null&&(rn.constructors[e]=[]),rn.constructors[e].push(t)}static checkForDuplicate(e){for(const t in rn.constructors)rn.constructors[+t].forEach(r=>{if(r===e)throw new q("Duplicate callback constructor.")})}static clear(){rn.constructors={}}static createCallbacks(e){const t=[];for(const n in rn.constructors){const r=+n;e>=r&&t.push(...rn.constructors[r])}return t.map(n=>new n)}}rn.constructors={};function gg(s,e,t,n,r,i,o,a,l){const u=new v3,c=[new _3,...rn.createCallbacks(e)];s!=null&&c.push(...s),c.push(u);const h=new x3(c);return h.setParams({epochs:t,initialEpoch:n,samples:r,steps:i,batchSize:o,verbose:e,doValidation:a,metrics:l}),{callbackList:h,history:u}}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function yg(s,e={},t=!1){return Oo(s,on.getMap().classNameMap,e,"layer",t)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function bg(s,e){return Q(()=>{const t=ne(.5,Lp(e)),n=im($o(e,t),s.dtype);return ut(kr(s,n),-1)})}function wg(s,e){return Q(()=>im(kr(Wa(s,-1),Wa(e,-1)),"float32"))}function k3(s,e){return Q(()=>De(Ae(Ol(kr(s,1),kr(e,1))),"float32"))}function I3(s,e){return Q(()=>De(Ae(Ol(kr(s,0),kr(e,1))),"float32"))}function E3(s,e){return Q(()=>{const t=k3(s,e),n=I3(s,e),r=he(t,n);return De(mr($o(r,0),_e(t,r),0),"float32")})}function T3(s,e){return Bl(s,e)}function A3(s,e){return s.rank===e.rank&&(s=Ml(s,[s.rank-1])),e=Wa(e,-1),e.dtype!==s.dtype&&(e=De(e,s.dtype)),De(kr(s,e),"float32")}const C3=Ll,N3=Ll,$3=Yc,D3=Yc,O3=Zc,M3=Zc,xg=ko,P3=ag,_g=sl,il={binaryAccuracy:bg,categoricalAccuracy:wg,precision:E3,categoricalCrossentropy:xg,sparseCategoricalCrossentropy:_g,mse:C3,MSE:N3,mae:$3,MAE:D3,mape:O3,MAPE:M3,cosine:P3};function R3(s){if(typeof s=="string"&&s in il)return il[s];if(typeof s!="string"&&s!=null)return s;throw new q(`Unknown metric ${s}`)}function ea(s){if(Xn(s!==null,`Unknown LossOrMetricFn ${s}`),typeof s=="string")return s;{let e;for(const t of Object.keys(rl))if(rl[t]===s){e=t;break}if(e!==void 0)return e;for(const t of Object.keys(il))if(il[t]===s){e=t;break}return e!==void 0?e:s.name}}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function L3(s){const e={Adagrad:()=>$r.adagrad(.01),Adadelta:()=>$r.adadelta(1,.95,Ye()),Adam:()=>$r.adam(.001,.9,.999,Ye()),Adamax:()=>$r.adamax(.002,.9,.999,Ye(),0),RMSProp:()=>$r.rmsprop(.001,.9,0,Ye()),SGD:()=>$r.sgd(.01)};if(e.adagrad=e.Adagrad,e.adadelta=e.Adadelta,e.adam=e.Adam,e.adamax=e.Adamax,e.rmsprop=e.RMSProp,e.sgd=e.SGD,s in e)return e[s]();throw new q(`Unknown Optimizer ${s}`)}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */const Sf=1*1024*1024;function kf(s,e,t=!1){if(s==null||typeof s!="object"||Object.getPrototypeOf(s)!==Object.prototype||!Wu(s))throw new Error("User-defined metadata is expected to be a JSON object, but is not.");if(t){const n=JSON.stringify(s);n.length>Sf&&console.warn(`User-defined metadata of model "${e}" is too large in size (length=${n.length} when serialized). It is not recommended to store such large objects in user-defined metadata. Please make sure its serialized length is <= ${Sf}.`)}}function Wu(s){if(s===null)return!0;if(typeof s=="object")if(Object.getPrototypeOf(s)===Object.prototype){const e=Object.keys(s);for(const t of e)if(typeof t!="string"||!Wu(s[t]))return!1;return!0}else if(Array.isArray(s)){for(const e of s)if(!Wu(e))return!1;return!0}else return!1;else{const e=typeof s;return e==="string"||e==="number"||e==="boolean"}}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function B3(s,e,t,n=console.log){const r=U3(s),i=["Layer (type)","Input Shape","Output shape","Param #"];r?(e=e||90,t=t||[.32,.61,.89,1]):(e=e||115,t=t||[.24,.48,.7,.8,1]),t[t.length-1]<=1&&(t=t.map(c=>Math.floor(e*c)));let o;if(!r){i.push("Receives inputs"),o=[];for(const c in s.nodesByDepth)o.push(...s.nodesByDepth[c])}n("_".repeat(e)),ol(i,t,n),n("=".repeat(e));const a=s.layers;for(let c=0;c<a.length;++c)r?z3(a[c],t,n):W3(a[c],t,o,n),n((c===a.length-1?"=":"_").repeat(e));s.checkTrainableWeightsConsistency();const l=F3(s),u=tl(s.nonTrainableWeights);n(`Total params: ${l+u}`),n(`Trainable params: ${l}`),n(`Non-trainable params: ${u}`),n("_".repeat(e))}function F3(s){let e;return s.collectedTrainableWeights!=null?e=tl(s.collectedTrainableWeights):e=tl(s.trainableWeights),e}function U3(s){let e=!0;const t=[],n=[];for(const r in s.nodesByDepth)t.push(s.nodesByDepth[r]);for(const r of t){if(r.length>1||r.length===1&&r[0].inboundLayers.length>1){e=!1;break}n.push(...r)}if(e)for(const r of s.layers){let i=!1;for(const o of r.inboundNodes)if(n.indexOf(o)!==-1)if(i){e=!1;break}else i=!0;if(!e)break}return e}function ol(s,e,t=console.log){let n="";for(let r=0;r<s.length;++r)r>0&&(n=n.slice(0,n.length-1)+" "),n+=s[r],n=n.slice(0,e[r]),n+=" ".repeat(e[r]-n.length);t(n)}function z3(s,e,t){let n,r;try{r=s.inboundNodes.map(l=>JSON.stringify(l.inputShapes)).join(",")}catch{r="multiple"}try{n=JSON.stringify(s.outputShape)}catch{n="multiple"}const i=s.name,o=s.getClassName(),a=[`${i} (${o})`,r,n,s.countParams().toString()];ol(a,e,t)}function W3(s,e,t,n){let r,i;try{i=s.inboundNodes.map(h=>JSON.stringify(h.inputShapes)).join(",")}catch{i="multiple"}try{r=JSON.stringify(s.outputShape)}catch{r="multiple"}const o=[];for(const h of s.inboundNodes)if(!(t!=null&&t.length>0&&t.indexOf(h)===-1))for(let d=0;d<h.inboundLayers.length;++d){const w=h.inboundLayers[d].name,k=h.nodeIndices[d],A=h.tensorIndices[d];o.push(`${w}[${k}][${A}]`)}const a=s.name,l=s.getClassName(),u=o.length===0?"":o[0],c=[`${a} (${l})`,i,r,s.countParams().toString(),u];ol(c,e,n);for(let h=1;h<o.length;++h)ol(["","","","",o[h]],e,n)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function vg(s,e,t){return(s==="inboundNodes"||s==="outputLayers"||s==="inputLayers")&&e===0&&typeof t=="string"}function Gu(s,e){if(s===null)return null;if(typeof s=="string")return nr(s);if(typeof s=="number"||typeof s=="boolean")return s;if(s instanceof Array){const t=[],n=s.length;for(let r=0;r<n;++r){const i=s[r];vg(e,r,i)?t.push(i):t.push(Gu(i,e))}return t}else{const t={};for(const n of Object.keys(s)){const r=s[n];if(n==="name"&&typeof r=="string")t[n]=r;else{const i=nr(n);t[i]=Gu(r,i)}}return t}}function Vu(s,e){if(s==null)return null;if(typeof s=="string")return us(s);if(typeof s=="number"||typeof s=="boolean")return s;if(s instanceof Array){const t=[],n=s.length;for(let r=0;r<n;++r){const i=s[r];vg(e,r,i)?t.push(i):t.push(Vu(i,e))}return t}else{const t={};for(const n of Object.keys(s)){const r=s[n],i=us(n);(n==="name"||n==="className")&&typeof r=="string"?t[i]=r:t[i]=Vu(r,n)}return t}}/** @license See the LICENSE file. */const Sg="4.20.0";/**
 * @license
 * Copyright 2022 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */class kg{constructor(e){this.maxEntries=e||100,this.cache=new Map}get(e){let t;return this.cache.has(e)&&(t=this.cache.get(e),this.cache.delete(e),this.cache.set(e,t)),t}put(e,t){if(this.cache.has(e))this.cache.delete(e);else if(this.cache.size>=this.maxEntries){const n=this.cache.keys().next().value;this.cache.delete(n)}this.cache.set(e,t)}getMaxEntries(){return this.maxEntries}setMaxEntries(e){if(e<0)throw new Error(`The maxEntries of LRU caches must be at least 0, but got ${e}.`);if(this.maxEntries>e)for(let t=0;t<this.maxEntries-e;t++){const n=this.cache.keys().next().value;this.cache.delete(n)}this.maxEntries=e}}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */class Bo extends $n{constructor(e){if(super({dtype:e.dtype,name:e.name!=null?e.name:Fc("input").toString()}),e.batchSize==null&&(e.batchSize=null),e.sparse==null&&(e.sparse=!1),this.trainable=!1,this.built=!0,this.sparse=e.sparse,e.inputShape!=null&&e.batchInputShape!=null)throw new q("Only provide the inputShape OR batchInputShape argument to inputLayer, not both at the same time.");let t=e.batchInputShape;if(t==null){if(e.inputShape==null)throw new q("An InputLayer should be passed either a `batchInputShape` or an `inputShape`.");t=[e.batchSize].concat(e.inputShape)}else if(e.batchSize!=null)throw new q("Cannot specify batchSize if batchInputShape is specified when creating an InputLayer.");const n=e.dtype||"float32";this.batchInputShape=t,this.dtype=n,this.inputSpec=[{shape:t}];const r=new Er(this.dtype,this.batchInputShape,this,[],{},this.name);r.nodeIndex=0,r.tensorIndex=0,new Hc({outboundLayer:this,inboundLayers:[],nodeIndices:[],tensorIndices:[],inputTensors:[r],outputTensors:[r],inputMasks:[null],outputMasks:[null],inputShapes:[t],outputShapes:[t]})}apply(e,t){throw new q(`Cannot pass any input to an InputLayer's apply() method. InputLayer name: ${this.name}`)}dispose(){return{refCountAfterDispose:this._refCount,numDisposedVariables:0}}getConfig(){return{batchInputShape:this.batchInputShape,dtype:this.dtype,sparse:this.sparse,name:this.name}}}Bo.className="InputLayer";le(Bo);function G3(s){if(s.batchShape==null&&s.shape==null)throw new Error("Please provide to Input either a `shape` or a `batchShape` argument. Note that `shape` does not include the batch dimension.");if(s.batchShape!=null&&s.shape!=null)throw new q("Please provide either a `shape` or `batchShape` argument to Input, but not both.");let e=s.batchShape;s.shape!=null&&e==null&&(e=[null].concat(s.shape));let t=s.dtype;return t==null&&(t="float32"),new Bo({batchInputShape:e,name:s.name,dtype:t,sparse:s.sparse}).inboundNodes[0].outputTensors[0]}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function V3(s,e){if(s.dtype==null||s.dtype===e.dtype)return e;try{return De(e,s.dtype)}catch{throw new q(`The dtype of the feed (${e.dtype}) can not be cast to the dtype of the key '${s.name}' (${s.dtype}).`)}}class Cs{constructor(e){if(this.id2Value={},this.id2Mask={},this.name2Id={},e instanceof Cs)for(const t in e.id2Value)this.id2Value[t]=e.id2Value[t],t in e.id2Mask&&(this.id2Mask[t]=e.id2Mask[t]);else{if(e==null)return;for(const t of e)this.add(t.key,t.value)}}add(e,t,n){if(this.id2Value[e.id]==null)this.id2Value[e.id]=V3(e,t),this.name2Id[e.name]=e.id,n!=null&&(this.id2Mask[e.id]=n);else throw new q(`Duplicate key: name=${e.name}, id=${e.id}`);return this}addFeed(e){this.add(e.key,e.value)}hasKey(e){return this.id2Value[e.id]!=null}names(){return Object.keys(this.name2Id)}getValue(e){if(e instanceof Er){if(this.id2Value[e.id]==null)throw new q(`Nonexistent key: ${e.name}`);return this.id2Value[e.id]}else{const t=this.name2Id[e];if(t==null)throw new q(`Feed dict has no SymbolicTensor name: ${e}`);return this.id2Value[t]}}getMask(e){if(e instanceof Er){if(this.id2Value[e.id]==null)throw new q(`Nonexistent key: ${e.name}`);return this.id2Mask[e.id]}else{const t=this.name2Id[e];if(t==null)throw new q(`Feed dict has no SymbolicTensor name: ${e}`);return this.id2Mask[t]}}disposeMasks(){this.id2Mask!=null&&Pe(this.id2Mask)}}const If=new kg,Ef=new kg;function Ui(s,e,t,n){const r=t==null?!1:t.training,i=Array.isArray(s),o=i?s:[s],a=o.map(k=>k.name),l=[],u=e.names();for(const k of a)u.indexOf(k)!==-1?l.push(e.getValue(k)):l.push(null);const c=a.join(",")+"|"+e.names().sort().join(",");let h=If.get(c),d;if(h==null){const k=q3(o,e);h=k.sorted,d=k.recipientCounts,If.put(c,h),Ef.put(c,d)}d={},r||Object.assign(d,Ef.get(c));const w=new Cs(e);for(let k=0;k<h.length;++k){const A=h[k],m=A.sourceLayer;if(m instanceof Bo)continue;const S=[],b=[],f=[];let v=!1;for(const $ of A.inputs){const C=w.getValue($),g=w.getMask($);S.push(C),b.push(g),g!=null&&(v=!0),r||(d[$.name]--,d[$.name]===0&&!e.hasKey($)&&a.indexOf($.name)===-1&&!C.isDisposed&&$.sourceLayer.stateful!==!0&&f.push(C))}v&&(t=t||{},t.mask=b[0]);const _=Ne(m.apply(S,t));let E=null;m.supportsMasking&&(E=m.computeMask(S,b));const D=j3(A),M=Array.isArray(D)?D:[D];for(let $=0;$<M.length;++$){w.hasKey(M[$])||w.add(M[$],_[$],Array.isArray(E)?E[0]:E);const C=a.indexOf(M[$].name);C!==-1&&(l[C]=_[$])}r||Pe(f)}return w.disposeMasks(),i?l:l[0]}function q3(s,e){R(s!=null&&s.length>0,()=>"Expected at least one fetch, got none");let t=[],n={};if(s.length===1){const r=Tf(s[0],e);t=r.sorted,n=r.recipientMap}else{const r=new Set;for(const i of s){const{sorted:o,recipientMap:a}=Tf(i,e);for(const l of o)r.has(l.name)||(t.push(l),r.add(l.name));for(const l in a)n[l]==null&&(n[l]=new Set),a[l].forEach(u=>n[l].add(u))}}return{sorted:t,recipientCounts:H3(n)}}function H3(s){const e={};for(const t in s)e[t]=s[t].size;return e}function Tf(s,e){const t=new Set,n=[],r={};for(const a of e.names())t.add(a);const i=[],o=[];for(i.push(s);i.length>0;){const a=i[i.length-1];if(t.has(a.name)){i.pop();continue}const l=o[o.length-1]===i.length-1;if(a.inputs.length===0||l)i.pop(),n.push(a),t.add(a.name),l&&o.pop();else{o.push(i.length-1);for(const u of a.inputs)r[u.name]==null&&(r[u.name]=new Set),r[u.name].add(a.name),!t.has(u.name)&&i.push(u)}}return{sorted:n,recipientMap:r}}function j3(s){let e;if(s.sourceLayer.inboundNodes.length===1)e=s.sourceLayer.output;else{let t=null;for(let n=0;n<s.sourceLayer.inboundNodes.length;++n)for(const r of s.sourceLayer.inboundNodes[n].outputTensors)if(r.id===s.id){t=n;break}e=s.sourceLayer.getOutputAt(t)}return e}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */const K3=s=>{const e=Object.keys(s);if(e.length===0)return!1;const t=e[0].split("/");return!isNaN(parseInt(t[t.length-1],10))};class xn extends $n{constructor(e){if(super({}),this.containerNodes=new Set,this.name=e.name,this.name==null){const b=this.getClassName().toLowerCase();this.name=Fc(b)}if(this.supportsMasking=!1,this.trainable_=!0,Array.isArray(e.inputs)?this.inputs=e.inputs.slice():this.inputs=[e.inputs],Array.isArray(e.outputs)?this.outputs=e.outputs.slice():this.outputs=[e.outputs],gr(this.inputs).length!==this.inputs.length)throw new q(`The list of inputs passed to the model is redundant. All inputs should only appear once. Found: ${this.inputs.map(b=>b.name)}`);gr(this.outputs).length!==this.outputs.length&&console.warn(`The list of outputs passed to the model is redundant. All outputs should only appear once. Found: ${this.outputs.map(b=>b.name)}`),this.inputLayers=[],this.inputLayersNodeIndices=[],this.inputLayersTensorIndices=[],this.outputLayers=[],this.outputLayersNodeIndices=[],this.outputLayersTensorIndices=[],this.layers=[],this.internalContainerRefs=[];for(const b of this.outputs){const f=b.sourceLayer,v=b.nodeIndex,_=b.tensorIndex;this.outputLayers.push(f),this.outputLayersNodeIndices.push(v),this.outputLayersTensorIndices.push(_)}for(const b of this.inputs){const f=b.sourceLayer,v=b.nodeIndex,_=b.tensorIndex;Xn(v===0,"input layer has >1 nodes"),Xn(_===0,"input layer has >1 tensors"),this.inputLayers.push(f),this.inputLayersNodeIndices.push(v),this.inputLayersTensorIndices.push(_)}this.inputNames=[],this.outputNames=[],this.feedInputShapes=[],this.feedInputNames=[],this.feedOutputNames=[];for(let b=0;b<this.inputLayers.length;b++){const f=this.inputLayers[b];if(!(f instanceof Bo))throw new TypeError(`Input layers to a LayersModel must be InputLayer objects. Received inputs: ${e.inputs}. Input ${b} (0-based) originates from layer type ${f.getClassName()}.`);this.inputNames.push(f.name),this.feedInputShapes.push(f.batchInputShape),this.feedInputNames.push(f.name)}for(const b of this.outputLayers)this.outputNames.push(b.name);this.internalInputShapes=this.inputs.map(b=>b.shape),this.internalOutputShapes=this.outputs.map(b=>b.shape);const t={},n={},r={},i={},o={},a=[],l=(b,f,v,_,E,D)=>{(_==null||E==null||D==null)&&(_=b.sourceLayer,E=b.nodeIndex,D=b.tensorIndex);const M=_.inboundNodes[E];if(v.indexOf(M)!==-1)throw new Ms(`The tensor ${b.name} at layer "${_.name}" is part of a cycle.`);if(f.indexOf(M)!==-1)return;this.containerNodes.add(xn.nodeKey(_,E)),_.id in o||(o[_.id]=Object.keys(o).length),v.indexOf(M)===-1&&v.push(M);const $=M.inboundLayers.length;for(let C=0;C<$;C++){const g=M.inputTensors[C],p=M.inboundLayers[C],y=M.nodeIndices[C],x=M.tensorIndices[C];l(g,f,v,p,y,x)}for(f.push(M);v.indexOf(M)>=0;)v.splice(v.indexOf(M),1);a.push(M)},u=[],c=[];for(const b of this.outputs)l(b,u,c);const h=a.slice().reverse();for(const b of h){n[b.id]=b,b.id in t||(t[b.id]=0);let f=t[b.id];const v=r[b.outboundLayer.id]==null?0:r[b.outboundLayer.id];f=Math.max(f,v),r[b.outboundLayer.id]=f,i[b.outboundLayer.id]=b.outboundLayer,t[b.id]=f;for(let _=0;_<b.inboundLayers.length;_++){const E=b.inboundLayers[_],D=b.nodeIndices[_],M=E.inboundNodes[D],$=t[M.id]==null?0:t[M.id];t[M.id]=Math.max(f+1,$),n[M.id]=M}}const d={};for(const b in t){const f=t[b];f in d||(d[f]=[]),d[f].push(n[b])}const w={};for(const b in r){const f=r[b];f in w||(w[f]=[]),w[f].push(i[b])}let k=Object.keys(w).map(b=>parseInt(b,10)).sort(Zo);this.layers=[];for(const b of k){const f=w[b];f.sort((v,_)=>{const E=o[v.id],D=o[_.id];return E<D?-1:E>D?1:0});for(const v of f)v instanceof xn&&this.internalContainerRefs.push(v),this.layers.push(v)}this.layersByDepth=w,k=Object.keys(d).map(b=>parseInt(b,10)).sort(Zo);const A=this.inputs.slice(),m=[];for(const b of k)for(const f of d[b]){const v=f.outboundLayer;if(v!=null){for(const _ of f.inputTensors)if(A.indexOf(_)===-1)throw new Ms(`Graph disconnected: cannot obtain value for tensor ${_} at layer "${v.name}". The following previous layers were accessed without issue: ${m}`);for(const _ of f.outputTensors)A.push(_);m.push(v.name)}}this.nodesByDepth=d;const S=this.layers.map(b=>b.name);for(const b of S){const f=S.filter(v=>v===b).length;if(f!==1)throw new Ms(`The name "${b}" is used ${f} times in the model. All layer names should be unique. Layer names: `+JSON.stringify(S))}this.outboundNodes=[],this.inboundNodes=[],new Hc({outboundLayer:this,inboundLayers:[],nodeIndices:[],tensorIndices:[],inputTensors:this.inputs,outputTensors:this.outputs,inputMasks:this.inputs.map(b=>null),outputMasks:this.outputs.map(b=>null),inputShapes:this.inputs.map(b=>b.shape),outputShapes:this.outputs.map(b=>b.shape)}),this.built=!0,this._refCount=1}assertNotDisposed(){if(this._refCount===0)throw new Error(`Container '${this.name}' is already disposed.`)}dispose(){this.assertNotDisposed();const e={refCountAfterDispose:null,numDisposedVariables:0};if(--this._refCount===0){for(const t of this.layers)e.numDisposedVariables+=t.dispose().numDisposedVariables;for(const t of this.internalContainerRefs)e.numDisposedVariables+=t.dispose().numDisposedVariables}return e.refCountAfterDispose=this._refCount,e}get trainable(){return this.trainable_}set trainable(e){this.layers.forEach(t=>{t._trainableWeights.forEach(n=>n.trainable=e)}),this.trainable_=e}get trainableWeights(){if(this._trainableWeights.length>0)throw new q("Container instance unexpectedly contains _trainableWeights.The trainable weights of a Container are a union of the trainable weights of its consituent Layers. Its own _trainableWeights must remain an empty Array.");if(!this.trainable)return[];let e=[];for(const t of this.layers)e=e.concat(t.trainableWeights);return e}get nonTrainableWeights(){const e=[];for(const t of this.layers)e.push(...t.nonTrainableWeights);if(!this.trainable){const t=[];for(const n of this.layers)t.push(...n.trainableWeights);return t.concat(e)}return e}get weights(){return this.trainableWeights.concat(this.nonTrainableWeights)}loadWeights(e,t=!0){const n={};let r=0;const i=K3(e);i&&this.parseWeights(e);for(const a of this.layers)for(const[l,u]of a.weights.entries()){const c=i?`${u.name.split("/").slice(0,-1).join("/")+"/"}${l}`:u.originalName;if(n[c]!=null)throw new q(`Duplicate weight name: ${c}`);n[c]=u,r++}const o=[];for(const a in e){let l=a;if(n[a]==null){const u=a.split("/");l=u.slice(0,-2).concat([u[u.length-1]]).join("/")}if(n[l]!=null)o.push([n[l],e[a]]);else if(t)throw new q(`Provided weight data has no target variable: ${a}`);delete n[l]}if(t){const a=[];for(const l in n)a.push(l);if(a.length>0)throw new q(`${a.length} of ${r} weights are not set: ${a}`)}Rm(o)}parseWeights(e){for(const t in Object.keys(e)){const n=t.split("/"),r=["vars","layer_checkpoint_dependencies"],i=n.map(o=>o.startsWith("_")?o.slice(1):o).filter(o=>!r.includes(o)).join("/");i!==t&&(e[i]=e[t],delete e[t])}}updatedConfig(){const e=this.getConfig(),t={};return t.className=this.getClassName(),t.config=e,t.kerasVersion=`tfjs-layers ${Sg}`,t.backend="TensorFlow.js",t}toJSON(e,t=!0){const n=Vu(this.updatedConfig());return t?JSON.stringify(n):n}call(e,t){return Q(()=>{e=Ne(e);const n=new Cs;for(let r=0;r<this.inputs.length;++r)n.add(this.inputs[r],e[r]);return Ui(this.outputs,n,t)})}computeMask(e,t){return Q(()=>{e=Ne(e);let n;return t==null?n=Xa(null,e.length):n=Ne(t),this.runInternalGraph(e,n)[1]})}computeOutputShape(e){const t=el(e);if(t.length!==this.inputLayers.length)throw new q(`Invalid inputShape argument ${e}: model has ${this.inputLayers.length} tensor inputs.`);const n={};for(let a=0;a<t.length;a++){const l=this.inputLayers[a],u=t[a],c=l.name+"_0_0";n[c]=u}const r=Object.keys(this.nodesByDepth).map(a=>parseInt(a,10)).sort(Zo);if(r.length>1)for(const a of r){const l=this.nodesByDepth[a];for(const u of l){const c=u.outboundLayer;if(this.inputLayers.map(A=>A.id).indexOf(c.id)!==-1)continue;const h=[];for(let A=0;A<u.inboundLayers.length;A++){const m=u.inboundLayers[A],S=u.nodeIndices[A],b=u.tensorIndices[A],f=`${m.name}_${S}_${b}`,v=n[f];h.push(v)}const d=c.computeOutputShape(Rt(h)),w=el(d),k=c.inboundNodes.indexOf(u);for(let A=0;A<w.length;A++){const m=`${c.name}_${k}_${A}`;n[m]=w[A]}}}const i=[],o=[];for(let a=0;a<this.outputLayers.length;a++){const l=this.outputLayers[a],u=this.outputLayersNodeIndices[a],c=this.outputLayersTensorIndices[a],h=`${l.name}_${u}_${c}`;o.push(h)}for(let a=0;a<o.length;a++){const l=o[a];Xn(l in n),i.push(n[l])}return Rt(i)}runInternalGraph(e,t){t==null&&(t=Xa(null,e.length));const n={};for(let l=0;l<this.inputs.length;++l){const u=this.inputs[l],c=e[l],h=t[l];n[u.id]=[c,h]}const r=Object.keys(this.nodesByDepth).map(l=>parseInt(l,10)).sort(Zo);for(const l of r){const u=this.nodesByDepth[l];for(const c of u){const h=c.outboundLayer,d=c.inputTensors,w=c.outputTensors,k=new Array;for(const A of d)A.id in n&&k.push(n[A.id]);if(k.length===d.length){let A={},m,S,b,f;if(c.callArgs!=null&&(A=c.callArgs),k.length===1){const[v,_]=k[0];A.mask==null&&(A.mask=_),b=Ne(h.call(v,A)),f=Ne(h.computeMask(v,_)),m=[v],S=[_]}else m=k.map(v=>v[0]),S=k.map(v=>v[1]),A.mask==null&&(A.mask=S),b=Ne(h.call(m,A)),f=Ne(h.computeMask(m,S));if(h.activityRegularizer)throw new Se("LayersModel invocation with concrete Tensor value(s) in the presence of activity regularizer(s) is not supported yet.");for(let v=0;v<w.length;++v){const _=w[v],E=b[v],D=f[v];n[_.id]=[E,D]}}}}const i=[],o=[],a=[];for(const l of this.outputs){Xn(l.id in n,`Could not compute output ${l.name} : ${l.id}`);const[u,c]=n[l.id];a.push(u.shape),i.push(u),o.push(c)}return[i,o,a]}buildNodeConversionMap(e){const t={};let n;for(const r of this.layers){n=r instanceof xn?1:0;for(let i=0;i<r.inboundNodes.length;i++){const o=xn.nodeKey(r,i);this.containerNodes.has(o)&&(t[o]=n,n+=1)}}return t}getLayer(e,t){if(t!=null)return this.findLayer(t);if(e==null)throw new q("Provide either a layer name or layer index");if(typeof e=="number")return this.findLayer(e);for(const n of this.layers)if(n.name===e)return n;throw new q(`No such layer: ${e}`)}findLayer(e){if(this.layers.length<=e)throw new q(`Was asked to retrieve layer at index ${e}, but model only has ${this.layers.length} layer(s).`);return this.layers[e]}calculateLosses(){return Q(()=>{const e=[];for(const t of this.layers)for(let n=0;n<t.inboundNodes.length;++n){const r=xn.nodeKey(t,n);this.containerNodes.has(r)&&e.push(...t.calculateLosses())}return e})}getConfig(){const e={name:this.name},t=this.buildNodeConversionMap(this.layers),n=[];for(const o of this.layers){const a=o.getClassName(),l=o.getConfig(),u=[];for(let h=0;h<o.inboundNodes.length;h++){const d=o.inboundNodes[h],w=xn.nodeKey(o,h);let k={};if(this.containerNodes.has(w)){if(d.callArgs)try{JSON.stringify(d.callArgs),k=d.callArgs}catch{console.warn(`Layer ${o.name} was passed non-serializable keyword arguments: ${d.callArgs}. They will not be included in the serialized model (and thus will be missing at deserialization time).`),k={}}if(d.inboundLayers.length>0){const A=[];for(let m=0;m<d.inboundLayers.length;m++){const S=d.inboundLayers[m],b=d.nodeIndices[m],f=d.tensorIndices[m],v=xn.nodeKey(S,b);let _=t[v];_==null&&(_=0),A.push([S.name,_,f,k])}u.push(A)}}}const c={};c.name=o.name,c.className=a,c.config=l,c.inboundNodes=u,n.push(c)}e.layers=n;const r=[];for(let o=0;o<this.inputLayers.length;o++){const a=this.inputLayers[o],l=this.inputLayersNodeIndices[o],u=xn.nodeKey(a,l);if(!this.containerNodes.has(u))continue;let c=t[u];c==null&&(c=0);const h=this.inputLayersTensorIndices[o];r.push([a.name,c,h])}e.inputLayers=r;const i=[];for(let o=0;o<this.outputLayers.length;o++){const a=this.outputLayers[o],l=this.outputLayersNodeIndices[o],u=xn.nodeKey(a,l);if(!this.containerNodes.has(u))continue;let c=t[u];c==null&&(c=0);const h=this.outputLayersTensorIndices[o];i.push([a.name,c,h])}return e.outputLayers=i,e}static fromConfig(e,t,n={},r=!1){const i={},o={};function a(m,S){m.name in o?o[m.name].push(S):o[m.name]=[S]}function l(m,S){const b=[];let f;for(const v of S){const _=v[0],E=v[1],D=v[2];if(f=v[3]==null?{}:v[3],!(_ in i)){a(m,S);return}const M=i[_];if(M.inboundNodes.length<=E){a(m,S);return}const $=M.inboundNodes[E];b.push($.outputTensors[D])}b.length>0&&m.apply(Rt(b),f)}function u(m){const S=m.name,b=yg(m,t.customObjects!=null?t.customObjects:{});b.setFastWeightInitDuringBuild(r),i[S]=b,m.inboundNodes.forEach(v=>{if(!(v instanceof Array))throw new q(`Corrupted configuration, expected array for nodeData: ${v}`);a(b,v)})}const c=t.name,h=t.layers;for(const m of h)u(m);for(;!$v(o);)for(const m of h){const S=i[m.name];if(S.name in o){const b=o[S.name];delete o[S.name];for(const f of b)l(S,f)}}const d=[],w=[],k=t.inputLayers;for(const m of k){const S=m[0],b=m[1],f=m[2];Xn(S in i);const _=i[S].inboundNodes[b].outputTensors;d.push(_[f])}const A=t.outputLayers;for(const m of A){const S=m[0],b=m[1],f=m[2];Xn(S in i);const _=i[S].inboundNodes[b].outputTensors;w.push(_[f])}return new e({inputs:d,outputs:w,name:c})}get stateful(){if(this._stateful)throw new q("Container instance unexpectedly has _stateful = true. The statefulness of a Container is determined by the Layers it contains. Its _stateful property must remain the default false.");for(const e of this.layers)if(e.stateful)return!0;return!1}resetStates(){Q(()=>{this.layers.forEach(e=>{e.stateful&&e.resetStates()})})}}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function X3(s,e,t){const n=e.length;if(s==null||Array.isArray(s)&&s.length===0)return e.map(r=>null);if(n===1)return Array.isArray(s)&&s.length===1?s:typeof s=="object"&&e[0]in s?[s[e[0]]]:[s];if(Array.isArray(s)){if(s.length!==n)throw new Error(`Provided ${t} is an array of ${s.length} element(s), but the model has ${n} outputs. Make sure a set of weights is provided for each model output.`);return s}else if(typeof s=="object"&&Object.keys(s).length>0&&typeof s[Object.keys(s)[0]]=="object"){const r=[];return e.forEach(i=>{i in s?r.push(s[i]):r.push(null)}),r}else throw new Error(`The model has multiple (${n}) outputs, so ${t} must be either an array with ${n} elements or an object with ${e} keys. Provided ${t} not understood: ${JSON.stringify(s)}`)}function Ig(s,e){return X3(s,e,"classWeight")}async function Eg(s,e,t,n){if(t!=null){const r=Q(()=>{if(s.shape.length===1)return dr(s);if(s.shape.length===2){if(s.shape[1]>1)return Wa(s,1);if(s.shape[1]===1)return ae(s,[s.shape[0]]);throw new Error(`Encountered unexpected last-dimension size (${s.shape[1]}) during handling of class weights. The size is expected to be >= 1.`)}else throw new Error(`Unexpected rank of target (y) tensor (${s.rank}) during handling of class weights. The rank is expected to be 1 or 2.`)}),i=Array.from(await r.data());Pe(r);const o=[];return i.forEach(a=>{if(t[a]==null)throw new Error(`classWeight must contain all classes in the training data. The class ${a} exists in the data but not in classWeight`);o.push(t[a])}),At(o,"float32")}else return null}function Y3(s,e){return ne(s,e)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */const Z3=32;function Tg(s,e){let t,n;const r=e;t=r.xs,n=r.ys,R(t!=null&&n!=null,()=>`A Dataset iterator for fitDataset() is expected to generate objects of the form \`{xs: xVal, ys: yVal}\`, where the two values may be \`tf.Tensor\`, an array of Tensors, or a map of string to Tensor.  The provided Dataset instead generates ${e}`);const i=Af("input",s.inputNames,t),o=Af("output",s.outputNames,n),a=i[0].shape[0];R(i.length===s.inputs.length,()=>`LayersModel has ${s.inputs.length} inputs, but the dataset provides ${i.length} inputs.  (Expected input keys: ${JSON.stringify(s.inputNames)})`),R(o.length===s.outputs.length,()=>`LayersModel has ${s.outputs.length} outputs, but the dataset provides ${o.length} outputs.  (Expected output keys: ${JSON.stringify(s.outputNames)})`);for(let l=0;l<i.length;l++)R(i[l].shape[0]===a,()=>`Batch size mismatch: input ${s.inputNames[l]} has ${i[l].shape[0]}; expected  ${a} based on input ${s.inputNames[0]}.`);for(let l=0;l<o.length;l++)R(o[l].shape[0]===a,()=>`Batch size mismatch: output ${s.outputNames[l]} has ${o[l].shape[0]}; expected  ${a} based on input ${s.inputNames[0]}.`);return{xs:i,ys:o}}function Af(s,e,t){if(t instanceof pt)return[t];if(Array.isArray(t))return R(t.length===e.length,()=>`Received an array of ${t.length} Tensors, but expected ${e.length} to match the ${s} keys ${e}.`),t;{const n=[];for(const r of e){if(t[r]==null)throw new q(`The feature data generated by the dataset lacks the required ${s} key '${r}'.`);n.push(t[r])}return n}}function Q3(s){if(s.length===3)throw new Se("Validation with sample weights is not implemented yet.");return{xs:s[0],ys:s[1]}}async function J3(s,e,t){const n=t.batchesPerEpoch!=null;if(R(s.optimizer!=null,()=>"You must compile a model before training/testing. Use LayersModel.compile(modelCompileConfig)."),R(t!=null,()=>"For fitDataset(), the 2nd argument (config) is required, but it is not provided in this call."),R(t.epochs!=null&&t.epochs>0&&Number.isInteger(t.epochs),()=>`For fitDataset(), config.epochs is expected to be a positive integer, but got ${t.epochs}`),R(!n||t.batchesPerEpoch>0&&Number.isInteger(t.batchesPerEpoch),()=>`For fitDataset(), config.batchesPerEpoch is expected to be a positive integer if specified, but got ${t.batchesPerEpoch}`),R(t.validationSplit==null,()=>"`validationSplit` is not supported by `fitDataset()`. Use validationData instead."),s.isTraining)throw new Error("Cannot start training because another fit() call is ongoing.");s.isTraining=!0;try{const r=t.validationData!=null;let i,o;if(r)if(Cf(t.validationData))R(t.validationBatches==null||t.validationBatches>0&&Number.isInteger(t.validationBatches),()=>`For fitDataset() with dataset-based validation, config.validationBatches is expected not to be provided, or to be a positive integer, but got ${t.validationBatches}`);else{const m=Q3(t.validationData);i=m.xs,o=m.ys}const a=s.makeTrainFunction(),l=s.getDedupedMetricsNames();let u;r?u=l.slice().concat(l.map(m=>"val_"+m)):u=l.slice();const c=mg(t.callbacks,t.yieldEvery),h=t.verbose==null?1:t.verbose,{callbackList:d,history:w}=gg(c,h,t.epochs,null,null,eS(e,t),null,r,u);d.setModel(s),s.history=w,await d.onTrainBegin(),s.stopTraining_=!1;let k=t.initialEpoch==null?0:t.initialEpoch,A=await e.iterator();for(;k<t.epochs;){const m={};await d.onEpochBegin(k);let S=0,b=0;for(n||(A=await e.iterator());!n||S<t.batchesPerEpoch;){const f=await A.next();if(n&&f.done){console.warn(`You provided \`batchesPerEpoch\` as ${t.batchesPerEpoch}, but your dataset iterator ran out of data after ${S} batches; interrupting training. Make sure that your dataset can generate at least \`batchesPerEpoch * epochs\` batches (in this case, ${t.batchesPerEpoch*t.epochs} batches). You may need to use the repeat() function when building your dataset.`);break}if(f.value!=null){const{xs:v,ys:_}=Tg(s,f.value),E={};E.batch=b,E.size=v[0].shape[0],await d.onBatchBegin(b,E);const D=[];if(t.classWeight!=null){const C=Ig(t.classWeight,s.outputNames);for(let g=0;g<C.length;++g)D.push(await Eg(_[g],null,C[g]))}const M=v.concat(_).concat(D),$=a(M);Pe(M);for(let C=0;C<l.length;++C){const g=l[C],p=$[C];E[g]=p,oi(p)}await d.onBatchEnd(b,E),pg(E),b++,S++}if(n?S>=t.batchesPerEpoch:f.done){if(r){let v;Cf(t.validationData)?v=Ne(await s.evaluateDataset(t.validationData,{batches:t.validationBatches})):v=Ne(s.evaluate(i,o,{batchSize:t.validationBatchSize==null?Z3:t.validationBatchSize,verbose:0}));for(let _=0;_<s.metricsNames.length;++_)m[`val_${s.metricsNames[_]}`]=v[_]}break}if(s.stopTraining_)break}if(await d.onEpochEnd(k,m),k++,s.stopTraining_)break}return await d.onTrainEnd(),await s.history.syncData(),s.history}finally{s.isTraining=!1}}function eS(s,e){let t=null;return e.batchesPerEpoch!=null?t=e.batchesPerEpoch:Number.isFinite(s.size)&&(t=s.size),t}function Cf(s){return typeof s.iterator=="function"}function tS(s){return typeof s.next=="function"}async function nS(s,e,t){t=t||{};const n=t.batches!=null,r=s.testFunction;let i=[];if(t.verbose>0)throw new Se("Verbose mode is not implemented yet.");R(!n||t.batches>0&&Number.isInteger(t.batches),()=>`Test loop expects \`batches\` to be a positive integer, but received ${JSON.stringify(t.batches)}`);const o=tS(e)?e:await e.iterator();let a=0,l=0;for(;!n||l<t.batches;){const u=await o.next();if(i=Q(()=>{if(u.value){const{xs:c,ys:h}=Tg(s,u.value),d=c.concat(h),w=Q(()=>r(d));if(Pe(d),l===0)for(let A=0;A<w.length;++A)i.push(Qt(0));const k=d[0].shape[0];for(let A=0;A<w.length;++A){const m=w[A],S=i[A];i[A]=Q(()=>he(i[A],ne(k,m))),l>0&&Pe(S)}Pe(w),a+=k,++l}return i}),u.done){n&&console.warn(`Your dataset iterator ran out of data during evaluateDataset(). Interrupting evalution. Make sure that your dataset can generate at least \`batches\` batches (in this case, ${t.batches} batches). You may need to use the repeat() function when building your dataset.`);break}}for(let u=0;u<i.length;++u){const c=i[u];i[u]=_e(i[u],a),Pe(c)}return Rt(i)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function uu(s){R(s>0&&Number.isInteger(s),()=>`batchSize is required to be a positive integer, but got ${s}`)}function Oi(s,e,t){return s==null?[null]:Array.isArray(s)?s.map(n=>yr(n,e,t-e)):yr(s,e,t-e)}function qu(s,e){return Q(()=>s==null?null:Array.isArray(s)?s.map(t=>qu(t,e)):Uv(s,e.dtype==="int32"?e:De(e,"int32")))}function cu(s,e){const t=[];let n=0,r=null;for(;n<s;)r=n+e,r>=s&&(r=s),t.push([n,r]),n=r;return t}function Ag(s){const e=[];s instanceof pt&&(s=[s]);for(let t=0;t<s.length;++t){const n=s[t];if(n.rank===1)e.push(Lc(n,1));else{if(n.rank===0)throw new Error("Expected tensor to be at least 1D, but received a 0D tensor (scalar).");e.push(n)}}return e}function pn(s,e){if(s==null)return;const t=[];if(e instanceof pt)t.push(e.id);else if(Array.isArray(e))e.forEach(r=>t.push(r.id));else if(e!=null)for(const r in e){const i=e[r];t.push(i.id)}const n=[];if(s instanceof pt)t.indexOf(s.id)===-1&&n.push(s);else if(Array.isArray(s))s.forEach(r=>{t.indexOf(r.id)===-1&&n.push(r)});else if(s!=null)for(const r in s){const i=s[r];t.indexOf(i.id)===-1&&n.push(i)}n.forEach(r=>{r.isDisposed||r.dispose()})}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function sS(s){return s instanceof pt}function Hu(s){return Array.isArray(s)}function Nf(s){return!sS(s)&&!Hu(s)}function $f(s,e,t,n=!0,r=""){if(e==null||e.length===0){if(s!=null){let o=!1;if(Hu(s)&&s.length>0)o=!0;else if(Nf(s)){for(const a in s)if(s.hasOwnProperty(a)){o=!0;break}}else o=!0;if(o)throw new q(`Error when checking model ${r} expected no data, but got ${s}`)}return[]}if(s==null)return e.map(o=>null);let i;if(Nf(s)){s=s,i=[];for(const o of e){if(s[o]==null)throw new q(`No data provided for "${o}". Need data for each key in: ${e}`);i.push(s[o])}}else if(Hu(s)){if(s=s,s.length!==e.length)throw new q(`Error when checking model ${r}: the Array of Tensors that you are passing to your model is not the size the model expected. Expected to see ${e.length} Tensor(s), but instead got the following list of Tensor(s): ${s}`);i=s}else{if(s=s,e.length>1)throw new q(`The model ${r} expects ${e.length} Tensor(s), but only received one Tensor. Found: Tensor with shape ${s.shape}`);i=[s]}if(i=Ag(i),t!=null)for(let o=0;o<e.length;++o){if(t[o]==null)continue;const a=i[o];if(a.shape.length!==t[o].length)throw new q(`Error when checking ${r}: expected ${e[o]} to have ${t[o].length} dimension(s). but got array with shape ${a.shape}`);for(let l=0;l<t[o].length;++l){if(l===0&&!n)continue;const u=a.shape[l],c=t[o][l];if(c!=null&&c>=0&&u!==c)throw new q(`${r} expected a batch of elements where each example has shape [${t[o].slice(1,t[o].length)}] (i.e.,tensor shape [*,${t[o].slice(1,t[o].length)}]) but the ${r} received an input with ${a.shape[0]} examples, each with shape [${a.shape.slice(1,a.shape.length)}] (tensor shape [${a.shape}])`)}}return i}function rS(s,e,t){const n=gr(s.map(i=>i.shape[0]));n.sort();const r=gr(e.map(i=>i.shape[0]));if(r.sort(),n.length>1)throw new q(`All input Tensors (x) should have the same number of samples. Got array shapes: ${JSON.stringify(s.map(i=>i.shape))}`);if(r.length>1)throw new q(`All target Tensors (y) should have the same number of samples. Got array shapes: ${JSON.stringify(e.map(i=>i.shape))}`);if(n.length>0&&r.length>0&&!cn(n,r))throw new q(`Input Tensors should have the same number of samples as target Tensors. Found ${n[0]} input sample(s) and ${r[0]} target sample(s).`)}function iS(s,e,t){const n=[Ll,Bl,ko];for(let r=0;r<s.length;++r){const i=s[r],o=e[r],a=t[r];if(o!=null){if(o===ko&&i.shape[i.shape.length-1]===1)throw new q(`You are passing a target array of shape ${i.shape} while using a loss 'categorical_crossentropy'. 'categorical_crossentropy'expects targets to be binary matrices (1s and 0s) of shape [samples, classes].`);if(n.indexOf(o)!==-1){const l=i.shape.slice(1),u=a.slice(1);for(let c=0;c<l.length;++c){const h=l[c],d=u[c];if(d!=null&&h!==d)throw new q(`A target Tensor with shape ${i.shape} was passed for an output of shape ${a}, while using a loss function that expects targets to have the same shape as the output.`)}}}}}function Df(s,e,t,n=!0,r=""){let i;if(Array.isArray(s)){if(s.length!==e.length)throw new q(`Error when checking model ${r}: the Array of Tensors that you are passing to your model is not the size the the model expected. Expected to see ${e.length} Tensor(s), but instead got ${s.length} Tensors(s).`);i=s}else{if(e.length>1)throw new q(`The model expects ${e.length} ${r} Tensors, but only received one Tensor. Found: array with shape ${JSON.stringify(s.shape)}.`);i=[s]}if(t!=null)for(let o=0;o<e.length;++o){if(t[o]==null)continue;const a=i[o];if(a.shape.length!==t[o].length)throw new q(`Error when checking ${r}: expected ${e[o]} to have ${t[o].length} dimension(s), but got array with shape ${JSON.stringify(a.shape)}`);for(let l=0;l<t[o].length;++l){if(l===0&&!n)continue;const u=a.shape[l],c=t[o][l];if(c!=null&&c!==u)throw new q(`Error when checking ${r}: expected ${e[o]} to have shape ${JSON.stringify(t[o])} but got array with shape ${JSON.stringify(a.shape)}.`)}}}function oS(s,e){if(s==null||Array.isArray(s)&&s.length===0)return e.map(n=>[]);let t;if(typeof s=="string"||typeof s=="function")t=[s];else if(Array.isArray(s)||typeof s=="object")t=s;else throw new TypeError(`Type of metrics argument not understood. Expected an string,function, Array, or Object, found: ${s}`);if(Array.isArray(t))return e.map(n=>t);{const n=[];for(const r of e){let i=t.hasOwnProperty(r)?t[r]:[];Array.isArray(i)||(i=[i]),n.push(i)}return n}}const aS="layers-model";class Fl extends xn{constructor(e){super(e),this.isTraining=!1}summary(e,t,n=console.log){if(!this.built)throw new q("This model has never been called, thus its weights have not been created yet. So no summary can be displayed. Build the model first (e.g., by calling it on some test data).");B3(this,e,t,n)}compile(e){if(e.loss==null&&(e.loss=[]),this.loss=e.loss,typeof e.optimizer=="string")this.optimizer_=L3(e.optimizer),this.isOptimizerOwned=!0;else{if(!(e.optimizer instanceof Bs))throw new q("User-defined optimizer must be an instance of tf.Optimizer.");this.optimizer_=e.optimizer,this.isOptimizerOwned=!1}let t=[];if(!Array.isArray(e.loss)&&typeof e.loss!="string"&&typeof e.loss!="function"){e.loss=e.loss;for(const o in e.loss)if(this.outputNames.indexOf(o)===-1)throw new q(`Unknown entry in loss dictionary: "${o}". Only expected the following keys: ${this.outputNames}`);for(const o of this.outputNames)e.loss[o]==null&&console.warn(`Output "${o}" is missing from loss dictionary. We assume this was done on purpose, and we will not be expecting data to be passed to ${o} during training`),t.push(lu(e.loss[o]))}else if(Array.isArray(e.loss)){if(e.loss.length!==this.outputs.length)throw new q(`When passing an Array as loss, it should have one entry per model output. The model has ${this.outputs.length} output(s), but you passed loss=${e.loss}.`);t=e.loss.map(a=>lu(a))}else{const o=lu(e.loss);this.outputs.forEach(a=>{t.push(o)})}this.lossFunctions=t,this.feedOutputNames=[],this.feedOutputShapes=[],this.feedLossFns=[];for(let o=0;o<this.outputs.length;++o){const a=this.internalOutputShapes[o],l=this.outputNames[o];this.feedOutputNames.push(l),this.feedOutputShapes.push(a),this.feedLossFns.push(this.lossFunctions[o])}const n=[];this.metrics=e.metrics,this.metricsNames=["loss"],this.metricsTensors=[],Ia("loss",()=>{for(let o=0;o<this.outputs.length;++o){if(n.indexOf(o)!==-1)continue;const a=this.lossFunctions[o];this.outputs.length>1&&(this.metricsTensors.push([a,o]),this.metricsNames.push(this.outputNames[o]+"_loss"))}});const r=oS(e.metrics,this.outputNames),i=(o,a,l)=>{this.outputNames.length>1&&(a=this.outputNames[o]+"_"+a),this.metricsNames.push(a),this.metricsTensors.push([l,o])};Ia("metric",()=>{for(let o=0;o<this.outputs.length;++o){if(n.indexOf(o)!==-1)continue;const a=r[o];(u=>{let h,d,w;for(const k of u){if(typeof k=="string"&&["accuracy","acc","crossentropy","ce"].indexOf(k)!==-1){const m=this.internalOutputShapes[o];m[m.length-1]===1||this.lossFunctions[o]===Bl?["accuracy","acc"].indexOf(k)!==-1?d=bg:["crossentropy","ce"].indexOf(k)!==-1&&(d=T3):this.lossFunctions[o]===sl?["accuracy","acc"].indexOf(k)!==-1?d=A3:["crossentropy","ce"].indexOf(k)!==-1&&(d=_g):["accuracy","acc"].indexOf(k)!==-1?d=wg:["crossentropy","ce"].indexOf(k)!==-1&&(d=xg);let S;["accuracy","acc"].indexOf(k)!==-1?S="acc":["crossentropy","ce"].indexOf(k)!==-1&&(S="ce"),w=d,h=""+S}else w=R3(k),h=""+ea(k);let A;Ia(h,()=>{A=w}),i(o,h,A)}})(a)}}),this.collectedTrainableWeights=this.trainableWeights}checkTrainableWeightsConsistency(){this.collectedTrainableWeights!=null&&this.trainableWeights.length!==this.collectedTrainableWeights.length&&console.warn("Discrepancy between trainableweights and collected trainable weights. Did you set `model.trainable` without calling `model.compile()` afterwards?")}evaluate(e,t,n={}){const r=n.batchSize==null?32:n.batchSize;uu(r);const o=this.standardizeUserDataXY(e,t,!0,r);try{const a=o[0].concat(o[1]);this.makeTestFunction();const l=this.testFunction,u=this.testLoop(l,a,r,n.verbose,n.steps);return Rt(u)}finally{pn(o[0],e),pn(o[1],t)}}async evaluateDataset(e,t){return this.makeTestFunction(),nS(this,e,t)}checkNumSamples(e,t,n,r="steps"){let i;if(n!=null){if(i=null,t!=null)throw new q(`If ${r} is set, batchSize must be null or undefined.Got batchSize = ${t}`)}else if(e!=null)Array.isArray(e)?i=e[0].shape[0]:i=e.shape[0];else throw new q(`Either the input data should have a defined shape, or ${r} shoud be specified.`);return i}execute(e,t){if(Array.isArray(t)&&t.length===0)throw new q("`outputs` is an empty Array, which is not allowed.");const n=Array.isArray(t),r=n?t:[t],i=this.retrieveSymbolicTensors(r),o=new Cs;if(e instanceof pt&&(e=[e]),Array.isArray(e)){if(e.length!==this.inputs.length)throw new q(`The number of inputs provided (${e.length}) does not match the number of inputs of this model (${this.inputs.length}).`);for(let l=0;l<this.inputs.length;++l)o.add(this.inputs[l],e[l])}else for(const l of this.inputs){const u=e[l.name];if(u==null)throw new q(`No value is provided for the model's input ${l.name}`);o.add(l,u)}const a=Ui(i,o);return n?a:a[0]}retrieveSymbolicTensors(e){const t=Xa(null,e.length);let n=e.length;for(const r of this.layers){const i=Array.isArray(r.output)?r.output:[r.output],o=i.map(a=>a.name);for(let a=0;a<e.length;++a){const l=o.indexOf(e[a]);if(l!==-1&&(t[a]=i[l],n--),n===0)break}if(n===0)break}if(n>0){const r=[];throw t.forEach((i,o)=>{i==null&&r.push(e[o])}),new q(`Cannot find SymbolicTensors for output name(s): ${JSON.stringify(r)}`)}return t}predictLoop(e,t=32,n=!1){return Q(()=>{const r=this.checkNumSamples(e);if(n)throw new Se("Verbose predictLoop() is not implemented yet.");const i=cu(r,t),o=this.outputs.map(a=>[]);for(let a=0;a<i.length;++a)Q(()=>{const u=i[a][0],c=i[a][1],h=Oi(e,u,c),d=[];if(Array.isArray(h))for(let k=0;k<h.length;++k)d.push({key:this.inputs[k],value:h[k]});else d.push({key:this.inputs[0],value:h});const w=new Cs(d);return Ui(this.outputs,w)}).forEach((u,c)=>o[c].push(u));return Rt(o.map(a=>pr(a,0)))})}predict(e,t={}){const n=Ag(e);Df(n,this.inputNames,this.feedInputShapes,!1);try{const r=t.batchSize==null?32:t.batchSize;return uu(r),this.predictLoop(n,r)}finally{pn(n,e)}}predictOnBatch(e){Df(e,this.inputNames,this.feedInputShapes,!0);const t=(Array.isArray(e)?e[0]:e).shape[0];return this.predictLoop(e,t)}standardizeUserDataXY(e,t,n=!0,r){if(this.optimizer_==null)throw new Ms("You must compile a model before training/testing. Use LayersModel.compile(modelCompileArgs).");const i=[];for(let o=0;o<this.feedOutputShapes.length;++o){const a=this.feedOutputShapes[o];this.feedLossFns[o]===sl?i.push(a.slice(0,a.length-1).concat([1])):i.push(a)}if(e=$f(e,this.feedInputNames,this.feedInputShapes,!1,"input"),t=$f(t,this.feedOutputNames,i,!1,"target"),rS(e,t),iS(t,this.feedLossFns,this.feedOutputShapes),this.stateful&&r!=null&&r>0&&e[0].shape[0]%r!==0)throw new q(`In a stateful network, you should only pass inputs with a number of samples that is divisible by the batch size ${r}. Found: ${e[0].shape[0]} sample(s).`);return[e,t]}async standardizeUserData(e,t,n,r,i=!0,o){const[a,l]=this.standardizeUserDataXY(e,t,i,o);if(n!=null)throw new Error("sample weight is not supported yet.");let u=null;if(r!=null){const c=Ig(r,this.outputNames);u=[];for(let h=0;h<c.length;++h)u.push(await Eg(l[h],null,c[h]))}return[a,l,u]}testLoop(e,t,n,r=0,i){return Q(()=>{const o=this.checkNumSamples(t,n,i,"steps"),a=[];if(r>0)throw new Se("Verbose mode is not implemented yet.");if(i!=null)throw new Se("steps mode in testLoop() is not implemented yet");{const l=cu(o,n),u=At(Ya(0,o));for(let c=0;c<l.length;++c){const h=l[c][0],d=l[c][1],w=yr(u,h,d-h),k=qu(t,w),A=e(k);if(c===0)for(let m=0;m<A.length;++m)a.push(Qt(0));for(let m=0;m<A.length;++m){const S=A[m];a[m]=he(a[m],ne(d-h,S))}}for(let c=0;c<a.length;++c)a[c]=_e(a[c],o)}return a})}getDedupedMetricsNames(){const e=this.metricsNames,t=[];for(let n=0;n<e.length;++n){const r=e[n];let i=r;if(hf(e,r)>1){const o=hf(e.slice(0,n),r);i+=`_${o}`}t.push(i)}return t}makeTrainFunction(){return e=>{const t=[],n=e.slice(0,this.inputs.length),r=e.slice(this.inputs.length,this.inputs.length+this.outputs.length),i=e.slice(this.inputs.length+this.outputs.length,this.inputs.length+this.outputs.length*2),o=[],a=()=>{const h=[];for(let A=0;A<this.inputs.length;++A)h.push({key:this.inputs[A],value:n[A]});const d=new Cs(h),w=Ui(this.outputs,d,{training:!0});let k;for(let A=0;A<this.lossFunctions.length;++A){const m=this.lossFunctions[A];let S=m(r[A],w[A]);i[A]!=null&&(S=Y3(S,i[A]));const b=ut(S);t.push(b),A===0?k=S:k=he(k,S)}for(let A=0;A<this.metricsTensors.length;++A){let m;if(this.outputs.length>1&&A<this.outputs.length)m=t[A];else{const S=this.metricsTensors[A][0],b=this.metricsTensors[A][1];m=ut(S(r[b],w[b]))}oi(m),o.push(m)}return k=ut(k),this.calculateLosses().forEach(A=>{k=he(k,A)}),k},l=this.collectedTrainableWeights.map(h=>h.read());return[this.optimizer_.minimize(a,!0,l)].concat(o)}}makeTestFunction(){this.testFunction=e=>Q(()=>{const t=[];let n;const r=e.slice(0,this.inputs.length),i=e.slice(this.inputs.length,this.inputs.length+this.outputs.length),o=[];for(let u=0;u<this.inputs.length;++u)o.push({key:this.inputs[u],value:r[u]});const a=new Cs(o),l=Ui(this.outputs,a);for(let u=0;u<this.lossFunctions.length;++u){const c=this.lossFunctions[u],h=ut(c(i[u],l[u]));u===0?n=h:n=he(n,h),t.push(n)}for(let u=0;u<this.metricsTensors.length;++u){const c=this.metricsTensors[u][0],h=this.metricsTensors[u][1],d=ut(c(i[h],l[h]));t.push(d)}return t})}async fit(e,t,n={}){if(this.isTraining)throw new Error("Cannot start training because another fit() call is ongoing.");this.isTraining=!0;let r,i,o,a,l,u,c,h,d;try{const w=n.batchSize==null?32:n.batchSize;uu(w);const A=await this.standardizeUserData(e,t,n.sampleWeight,n.classWeight,!1,w);r=A[0],i=A[1],d=A[2];let m=!1,S;if(n.validationData!=null&&n.validationData.length>0){if(m=!0,n.validationData.length===2)l=n.validationData[0],u=n.validationData[1];else throw n.validationData.length===3?new Se("validationData including sample weights is not supported yet."):new q(`When passing validation data, it must contain 2 (valX, valY) or 3 (valX, valY, valSampleWeight) items; ${n.validationData} is invalid.`);const C=await this.standardizeUserData(l,u,null,null,!0,w);c=C[0],h=C[1],S=c.concat(h)}else if(n.validationSplit!=null&&n.validationSplit>0&&n.validationSplit<1){m=!0;const $=Math.floor(r[0].shape[0]*(1-n.validationSplit)),C=r[0].shape[0];c=Oi(r,$,C),o=r,r=Oi(r,0,$),h=Oi(i,$,C),a=i,i=Oi(i,0,$),S=c.concat(h)}else n.validationSteps!=null&&(m=!0);const b=r.concat(i).concat(d);this.checkTrainableWeightsConsistency();const f=this.makeTrainFunction(),v=this.getDedupedMetricsNames();let _,E;m?(this.makeTestFunction(),_=this.testFunction,E=v.slice().concat(v.map($=>"val_"+$))):(_=null,S=[],E=v.slice());const D=mg(n.callbacks,n.yieldEvery);return await this.fitLoop(f,b,v,w,n.epochs,n.verbose,D,_,S,n.shuffle,E,n.initialEpoch,null,null)}finally{this.isTraining=!1,pn(r,e),pn(i,t),pn(o,e),pn(a,t),pn(c,l),pn(h,u),d!=null&&Pe(d)}}async fitLoop(e,t,n,r,i,o,a,l,u,c,h,d,w,k){r==null&&(r=32),i==null&&(i=1),c==null&&(c=!0),d==null&&(d=0);let A=!1;if(l!=null&&u!=null&&(A=!0),k!=null&&(A=!0,w==null))throw new q("Can only use `validationSteps` when doing step-wise training, i.e., `stepsPerEpoch` must be set.");const m=this.checkNumSamples(t,r,w,"steps_per_epoch");let S;m!=null&&(S=Ya(0,m)),o==null&&(o=1);const{callbackList:b,history:f}=gg(a,o,i,d,m,w,r,A,h);b.setModel(this),this.history=f,await b.onTrainBegin(),this.stopTraining_=!1;for(let v=d;v<i;++v){await b.onEpochBegin(v);const _={};if(w!=null)throw new Se("stepsPerEpoch mode is not implemented yet.");{if(c==="batch")throw new Se("batch shuffling is not implemneted yet");c&&f0(S);const E=At(S),D=cu(m,r);for(let M=0;M<D.length;++M){const $={};if(await b.onBatchBegin(M,$),Q(()=>{const C=D[M][0],g=D[M][1],p=yr(E,C,g-C);$.batch=M,$.size=g-C;const y=qu(t,p),x=e(y);for(let I=0;I<n.length;++I){const N=n[I],L=x[I];$[N]=L,oi(L)}if(M===D.length-1&&A){const I=this.testLoop(l,u,r);for(let N=0;N<n.length;++N){const L=n[N],W=I[N];oi(W),_["val_"+L]=W}}}),await b.onBatchEnd(M,$),pg($),this.stopTraining_)break}E.dispose()}if(await b.onEpochEnd(v,_),this.stopTraining_)break}return await b.onTrainEnd(),await this.history.syncData(),this.history}async fitDataset(e,t){return J3(this,e,t)}async trainOnBatch(e,t){const n=await this.standardizeUserData(e,t),r=n[0],i=n[1],a=this.makeTrainFunction()(r.concat(i)),l=[];for(const u of a){const c=await u.data();l.push(c[0])}return Pe(a),pn(n[0],e),pn(n[1],t),Rt(l)}getNamedWeights(e){const t=[],n=e!=null&&e.trainableOnly,r=n?this.trainableWeights:this.weights,i=this.getWeights(n);for(let o=0;o<r.length;++o)n&&!r[o].trainable||t.push({name:r[o].originalName,tensor:i[o]});return t}set stopTraining(e){this.stopTraining_=e}get stopTraining(){return this.stopTraining_}get optimizer(){return this.optimizer_}set optimizer(e){this.optimizer_!==e&&(this.optimizer_=e,this.isOptimizerOwned=!1)}dispose(){const e=super.dispose();if(e.refCountAfterDispose===0&&this.optimizer!=null&&this.isOptimizerOwned){const t=qh().numTensors;this.optimizer_.dispose(),e.numDisposedVariables+=t-qh().numTensors}return e}getLossIdentifiers(){let e;if(typeof this.loss=="string")e=us(this.loss);else if(Array.isArray(this.loss)){for(const t of this.loss)if(typeof t!="string")throw new Error("Serialization of non-string loss is not supported.");e=this.loss.map(t=>us(t))}else{const t=Object.keys(this.loss);e={};const n=this.loss;for(const r of t)if(typeof n[r]=="string")e[r]=us(n[r]);else throw new Error("Serialization of non-string loss is not supported.")}return e}getMetricIdentifiers(){if(typeof this.metrics=="string"||typeof this.metrics=="function")return[us(ea(this.metrics))];if(Array.isArray(this.metrics))return this.metrics.map(e=>us(ea(e)));{const e={};for(const t in this.metrics)e[t]=us(ea(this.metrics[t]));return e}}getTrainingConfig(){return{loss:this.getLossIdentifiers(),metrics:this.getMetricIdentifiers(),optimizer_config:{class_name:this.optimizer.getClassName(),config:this.optimizer.getConfig()}}}loadTrainingConfig(e){if(e.weighted_metrics!=null)throw new Error("Loading weight_metrics is not supported yet.");if(e.loss_weights!=null)throw new Error("Loading loss_weights is not supported yet.");if(e.sample_weight_mode!=null)throw new Error("Loading sample_weight_mode is not supported yet.");const t=Gu(e.optimizer_config),n=yg(t);let r;if(typeof e.loss=="string")r=nr(e.loss);else if(Array.isArray(e.loss))r=e.loss.map(o=>nr(o));else if(e.loss!=null){r={};for(const o in e.loss)r[o]=nr(e.loss[o])}let i;if(Array.isArray(e.metrics))i=e.metrics.map(o=>nr(o));else if(e.metrics!=null){i={};for(const o in e.metrics)i[o]=nr(e.metrics[o])}this.compile({loss:r,metrics:i,optimizer:n})}async save(e,t){if(typeof e=="string"){const u=G1(e);if(u.length===0)throw new q(`Cannot find any save handlers for URL '${e}'`);if(u.length>1)throw new q(`Found more than one (${u.length}) save handlers for URL '${e}'`);e=u[0]}if(e.save==null)throw new q("LayersModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");const n=await jh(this.getNamedWeights(t)),a={modelTopology:this.toJSON(null,!1),format:aS,generatedBy:`TensorFlow.js tfjs-layers v${Sg}`,convertedBy:null};if((t==null?!1:t.includeOptimizer)&&this.optimizer!=null){a.trainingConfig=this.getTrainingConfig();const u="optimizer",{data:c,specs:h}=await jh(await this.optimizer.getWeights(),u);n.specs.push(...h),n.data=W1([n.data,c])}return this.userDefinedMetadata!=null&&(kf(this.userDefinedMetadata,this.name,!0),a.userDefinedMetadata=this.userDefinedMetadata),a.weightData=n.data,a.weightSpecs=n.specs,e.save(a)}setUserDefinedMetadata(e){kf(e,this.name),this.userDefinedMetadata=e}getUserDefinedMetadata(){return this.userDefinedMetadata}}Fl.className="Model";le(Fl);class Cg extends Fl{}Cg.className="Functional";le(Cg);const lS="This is not an object",uS="This is not a Float16Array object",Of="This constructor is not a subclass of Float16Array",Ng="The constructor property value is not an object",cS="Species constructor didn't return TypedArray object",hS="Derived constructor created TypedArray object which was too small length",so="Attempting to access detached ArrayBuffer",ju="Cannot convert undefined or null to object",Ku="Cannot mix BigInt and other types, use explicit conversions",Mf="@@iterator property is not callable",Pf="Reduce of empty array with no initial value",fS="The comparison function must be either a function or undefined",hu="Offset is out of bounds";function Re(s){return(e,...t)=>Pt(s,e,t)}function Ti(s,e){return Re(di(s,e).get)}const{apply:Pt,construct:zi,defineProperty:Rf,get:fu,getOwnPropertyDescriptor:di,getPrototypeOf:Fo,has:Xu,ownKeys:$g,set:Lf,setPrototypeOf:Dg}=Reflect,dS=Proxy,{EPSILON:pS,MAX_SAFE_INTEGER:Bf,isFinite:Og,isNaN:pi}=Number,{iterator:es,species:mS,toStringTag:Jc,for:gS}=Symbol,mi=Object,{create:Ul,defineProperty:Uo,freeze:yS,is:Ff}=mi,Yu=mi.prototype,bS=Yu.__lookupGetter__?Re(Yu.__lookupGetter__):(s,e)=>{if(s==null)throw Be(ju);let t=mi(s);do{const n=di(t,e);if(n!==void 0)return ps(n,"get")?n.get:void 0}while((t=Fo(t))!==null)},ps=mi.hasOwn||Re(Yu.hasOwnProperty),Mg=Array,Pg=Mg.isArray,zl=Mg.prototype,wS=Re(zl.join),xS=Re(zl.push),_S=Re(zl.toLocaleString),eh=zl[es],vS=Re(eh),{abs:SS,trunc:Rg}=Math,Wl=ArrayBuffer,kS=Wl.isView,Lg=Wl.prototype,IS=Re(Lg.slice),ES=Ti(Lg,"byteLength"),Zu=typeof SharedArrayBuffer<"u"?SharedArrayBuffer:null,TS=Zu&&Ti(Zu.prototype,"byteLength"),th=Fo(Uint8Array),AS=th.from,ht=th.prototype,CS=ht[es],NS=Re(ht.keys),$S=Re(ht.values),DS=Re(ht.entries),OS=Re(ht.set),Uf=Re(ht.reverse),MS=Re(ht.fill),PS=Re(ht.copyWithin),zf=Re(ht.sort),Mi=Re(ht.slice),RS=Re(ht.subarray),rt=Ti(ht,"buffer"),Us=Ti(ht,"byteOffset"),Te=Ti(ht,"length"),Bg=Ti(ht,Jc),LS=Uint8Array,Xt=Uint16Array,Wf=(...s)=>Pt(AS,Xt,s),nh=Uint32Array,BS=Float32Array,Tr=Fo([][es]()),Gl=Re(Tr.next),FS=Re(function*(){}().next),US=Fo(Tr),Be=TypeError,du=RangeError,Fg=WeakSet,Ug=Fg.prototype,zS=Re(Ug.add),WS=Re(Ug.has),Vl=WeakMap,sh=Vl.prototype,al=Re(sh.get),GS=Re(sh.has),rh=Re(sh.set),zg=new Vl,VS=Ul(null,{next:{value:function(){const e=al(zg,this);return Gl(e)}},[es]:{value:function(){return this}}});function ta(s){if(s[es]===eh&&Tr.next===Gl)return s;const e=Ul(VS);return rh(zg,e,vS(s)),e}const Wg=new Vl,Gg=Ul(US,{next:{value:function(){const e=al(Wg,this);return FS(e)},writable:!0,configurable:!0}});for(const s of $g(Tr))s!=="next"&&Uo(Gg,s,di(Tr,s));function Gf(s){const e=Ul(Gg);return rh(Wg,e,s),e}function ll(s){return s!==null&&typeof s=="object"||typeof s=="function"}function Vf(s){return s!==null&&typeof s=="object"}function ul(s){return Bg(s)!==void 0}function Qu(s){const e=Bg(s);return e==="BigInt64Array"||e==="BigUint64Array"}function qS(s){try{return Pg(s)?!1:(ES(s),!0)}catch{return!1}}function Vg(s){if(Zu===null)return!1;try{return TS(s),!0}catch{return!1}}function HS(s){return qS(s)||Vg(s)}function qf(s){return Pg(s)?s[es]===eh&&Tr.next===Gl:!1}function jS(s){return ul(s)?s[es]===CS&&Tr.next===Gl:!1}function na(s){if(typeof s!="string")return!1;const e=+s;return s!==e+""||!Og(e)?!1:e===Rg(e)}const cl=gS("__Float16Array__");function KS(s){if(!Vf(s))return!1;const e=Fo(s);if(!Vf(e))return!1;const t=e.constructor;if(t===void 0)return!1;if(!ll(t))throw Be(Ng);return Xu(t,cl)}const Ju=1/pS;function XS(s){return s+Ju-Ju}const qg=6103515625e-14,YS=65504,Hg=.0009765625,Hf=Hg*qg,ZS=Hg*Ju;function QS(s){const e=+s;if(!Og(e)||e===0)return e;const t=e>0?1:-1,n=SS(e);if(n<qg)return t*XS(n/Hf)*Hf;const r=(1+ZS)*n,i=r-(r-n);return i>YS||pi(i)?t*(1/0):t*i}const jg=new Wl(4),Kg=new BS(jg),Xg=new nh(jg),gn=new Xt(512),yn=new LS(512);for(let s=0;s<256;++s){const e=s-127;e<-24?(gn[s]=0,gn[s|256]=32768,yn[s]=24,yn[s|256]=24):e<-14?(gn[s]=1024>>-e-14,gn[s|256]=1024>>-e-14|32768,yn[s]=-e-1,yn[s|256]=-e-1):e<=15?(gn[s]=e+15<<10,gn[s|256]=e+15<<10|32768,yn[s]=13,yn[s|256]=13):e<128?(gn[s]=31744,gn[s|256]=64512,yn[s]=24,yn[s|256]=24):(gn[s]=31744,gn[s|256]=64512,yn[s]=13,yn[s|256]=13)}function Gn(s){Kg[0]=QS(s);const e=Xg[0],t=e>>23&511;return gn[t]+((e&8388607)>>yn[t])}const ih=new nh(2048);for(let s=1;s<1024;++s){let e=s<<13,t=0;for(;(e&8388608)===0;)e<<=1,t-=8388608;e&=-8388609,t+=947912704,ih[s]=e|t}for(let s=1024;s<2048;++s)ih[s]=939524096+(s-1024<<13);const Ai=new nh(64);for(let s=1;s<31;++s)Ai[s]=s<<23;Ai[31]=1199570944;Ai[32]=2147483648;for(let s=33;s<63;++s)Ai[s]=2147483648+(s-32<<23);Ai[63]=3347054592;const Yg=new Xt(64);for(let s=1;s<64;++s)s!==32&&(Yg[s]=1024);function Ce(s){const e=s>>10;return Xg[0]=ih[Yg[e]+(s&1023)]+Ai[e],Kg[0]}function ss(s){const e=+s;return pi(e)||e===0?0:Rg(e)}function pu(s){const e=ss(s);return e<0?0:e<Bf?e:Bf}function sa(s,e){if(!ll(s))throw Be(lS);const t=s.constructor;if(t===void 0)return e;if(!ll(t))throw Be(Ng);const n=t[mS];return n??e}function ro(s){if(Vg(s))return!1;try{return IS(s,0,0),!1}catch{}return!0}function jf(s,e){const t=pi(s),n=pi(e);if(t&&n)return 0;if(t)return 1;if(n||s<e)return-1;if(s>e)return 1;if(s===0&&e===0){const r=Ff(s,0),i=Ff(e,0);if(!r&&i)return-1;if(r&&!i)return 1}return 0}const oh=2,hl=new Vl;function Xr(s){return GS(hl,s)||!kS(s)&&KS(s)}function Ie(s){if(!Xr(s))throw Be(uS)}function ra(s,e){const t=Xr(s),n=ul(s);if(!t&&!n)throw Be(cS);if(typeof e=="number"){let r;if(t){const i=ye(s);r=Te(i)}else r=Te(s);if(r<e)throw Be(hS)}if(Qu(s))throw Be(Ku)}function ye(s){const e=al(hl,s);if(e!==void 0){const r=rt(e);if(ro(r))throw Be(so);return e}const t=s.buffer;if(ro(t))throw Be(so);const n=zi(Le,[t,s.byteOffset,s.length],s.constructor);return al(hl,n)}function Kf(s){const e=Te(s),t=[];for(let n=0;n<e;++n)t[n]=Ce(s[n]);return t}const Zg=new Fg;for(const s of $g(ht)){if(s===Jc)continue;const e=di(ht,s);ps(e,"get")&&typeof e.get=="function"&&zS(Zg,e.get)}const JS=yS({get(s,e,t){return na(e)&&ps(s,e)?Ce(fu(s,e)):WS(Zg,bS(s,e))?fu(s,e):fu(s,e,t)},set(s,e,t,n){return na(e)&&ps(s,e)?Lf(s,e,Gn(t)):Lf(s,e,t,n)},getOwnPropertyDescriptor(s,e){if(na(e)&&ps(s,e)){const t=di(s,e);return t.value=Ce(t.value),t}return di(s,e)},defineProperty(s,e,t){return na(e)&&ps(s,e)&&ps(t,"value")&&(t.value=Gn(t.value)),Rf(s,e,t)}});class Le{constructor(e,t,n){let r;if(Xr(e))r=zi(Xt,[ye(e)],new.target);else if(ll(e)&&!HS(e)){let o,a;if(ul(e)){o=e,a=Te(e);const l=rt(e);if(ro(l))throw Be(so);if(Qu(e))throw Be(Ku);const u=new Wl(a*oh);r=zi(Xt,[u],new.target)}else{const l=e[es];if(l!=null&&typeof l!="function")throw Be(Mf);l!=null?qf(e)?(o=e,a=e.length):(o=[...e],a=o.length):(o=e,a=pu(o.length)),r=zi(Xt,[a],new.target)}for(let l=0;l<a;++l)r[l]=Gn(o[l])}else r=zi(Xt,arguments,new.target);const i=new dS(r,JS);return rh(hl,i,r),i}static from(e,...t){const n=this;if(!Xu(n,cl))throw Be(Of);if(n===Le){if(Xr(e)&&t.length===0){const c=ye(e),h=new Xt(rt(c),Us(c),Te(c));return new Le(rt(Mi(h)))}if(t.length===0)return new Le(rt(Wf(e,Gn)));const l=t[0],u=t[1];return new Le(rt(Wf(e,function(c,...h){return Gn(Pt(l,this,[c,...ta(h)]))},u)))}let r,i;const o=e[es];if(o!=null&&typeof o!="function")throw Be(Mf);if(o!=null)qf(e)?(r=e,i=e.length):jS(e)?(r=e,i=Te(e)):(r=[...e],i=r.length);else{if(e==null)throw Be(ju);r=mi(e),i=pu(r.length)}const a=new n(i);if(t.length===0)for(let l=0;l<i;++l)a[l]=r[l];else{const l=t[0],u=t[1];for(let c=0;c<i;++c)a[c]=Pt(l,u,[r[c],c])}return a}static of(...e){const t=this;if(!Xu(t,cl))throw Be(Of);const n=e.length;if(t===Le){const i=new Le(n),o=ye(i);for(let a=0;a<n;++a)o[a]=Gn(e[a]);return i}const r=new t(n);for(let i=0;i<n;++i)r[i]=e[i];return r}keys(){Ie(this);const e=ye(this);return NS(e)}values(){Ie(this);const e=ye(this);return Gf(function*(){for(const t of $S(e))yield Ce(t)}())}entries(){Ie(this);const e=ye(this);return Gf(function*(){for(const[t,n]of DS(e))yield[t,Ce(n)]}())}at(e){Ie(this);const t=ye(this),n=Te(t),r=ss(e),i=r>=0?r:n+r;if(!(i<0||i>=n))return Ce(t[i])}with(e,t){Ie(this);const n=ye(this),r=Te(n),i=ss(e),o=i>=0?i:r+i,a=+t;if(o<0||o>=r)throw du(hu);const l=new Xt(rt(n),Us(n),Te(n)),u=new Le(rt(Mi(l))),c=ye(u);return c[o]=Gn(a),u}map(e,...t){Ie(this);const n=ye(this),r=Te(n),i=t[0],o=sa(n,Le);if(o===Le){const l=new Le(r),u=ye(l);for(let c=0;c<r;++c){const h=Ce(n[c]);u[c]=Gn(Pt(e,i,[h,c,this]))}return l}const a=new o(r);ra(a,r);for(let l=0;l<r;++l){const u=Ce(n[l]);a[l]=Pt(e,i,[u,l,this])}return a}filter(e,...t){Ie(this);const n=ye(this),r=Te(n),i=t[0],o=[];for(let u=0;u<r;++u){const c=Ce(n[u]);Pt(e,i,[c,u,this])&&xS(o,c)}const a=sa(n,Le),l=new a(o);return ra(l),l}reduce(e,...t){Ie(this);const n=ye(this),r=Te(n);if(r===0&&t.length===0)throw Be(Pf);let i,o;t.length===0?(i=Ce(n[0]),o=1):(i=t[0],o=0);for(let a=o;a<r;++a)i=e(i,Ce(n[a]),a,this);return i}reduceRight(e,...t){Ie(this);const n=ye(this),r=Te(n);if(r===0&&t.length===0)throw Be(Pf);let i,o;t.length===0?(i=Ce(n[r-1]),o=r-2):(i=t[0],o=r-1);for(let a=o;a>=0;--a)i=e(i,Ce(n[a]),a,this);return i}forEach(e,...t){Ie(this);const n=ye(this),r=Te(n),i=t[0];for(let o=0;o<r;++o)Pt(e,i,[Ce(n[o]),o,this])}find(e,...t){Ie(this);const n=ye(this),r=Te(n),i=t[0];for(let o=0;o<r;++o){const a=Ce(n[o]);if(Pt(e,i,[a,o,this]))return a}}findIndex(e,...t){Ie(this);const n=ye(this),r=Te(n),i=t[0];for(let o=0;o<r;++o){const a=Ce(n[o]);if(Pt(e,i,[a,o,this]))return o}return-1}findLast(e,...t){Ie(this);const n=ye(this),r=Te(n),i=t[0];for(let o=r-1;o>=0;--o){const a=Ce(n[o]);if(Pt(e,i,[a,o,this]))return a}}findLastIndex(e,...t){Ie(this);const n=ye(this),r=Te(n),i=t[0];for(let o=r-1;o>=0;--o){const a=Ce(n[o]);if(Pt(e,i,[a,o,this]))return o}return-1}every(e,...t){Ie(this);const n=ye(this),r=Te(n),i=t[0];for(let o=0;o<r;++o)if(!Pt(e,i,[Ce(n[o]),o,this]))return!1;return!0}some(e,...t){Ie(this);const n=ye(this),r=Te(n),i=t[0];for(let o=0;o<r;++o)if(Pt(e,i,[Ce(n[o]),o,this]))return!0;return!1}set(e,...t){Ie(this);const n=ye(this),r=ss(t[0]);if(r<0)throw du(hu);if(e==null)throw Be(ju);if(Qu(e))throw Be(Ku);if(Xr(e))return OS(ye(this),ye(e),r);if(ul(e)){const l=rt(e);if(ro(l))throw Be(so)}const i=Te(n),o=mi(e),a=pu(o.length);if(r===1/0||a+r>i)throw du(hu);for(let l=0;l<a;++l)n[l+r]=Gn(o[l])}reverse(){Ie(this);const e=ye(this);return Uf(e),this}toReversed(){Ie(this);const e=ye(this),t=new Xt(rt(e),Us(e),Te(e)),n=new Le(rt(Mi(t))),r=ye(n);return Uf(r),n}fill(e,...t){Ie(this);const n=ye(this);return MS(n,Gn(e),...ta(t)),this}copyWithin(e,t,...n){Ie(this);const r=ye(this);return PS(r,e,t,...ta(n)),this}sort(e){Ie(this);const t=ye(this),n=e!==void 0?e:jf;return zf(t,(r,i)=>n(Ce(r),Ce(i))),this}toSorted(e){Ie(this);const t=ye(this);if(e!==void 0&&typeof e!="function")throw new Be(fS);const n=e!==void 0?e:jf,r=new Xt(rt(t),Us(t),Te(t)),i=new Le(rt(Mi(r))),o=ye(i);return zf(o,(a,l)=>n(Ce(a),Ce(l))),i}slice(e,t){Ie(this);const n=ye(this),r=sa(n,Le);if(r===Le){const k=new Xt(rt(n),Us(n),Te(n));return new Le(rt(Mi(k,e,t)))}const i=Te(n),o=ss(e),a=t===void 0?i:ss(t);let l;o===-1/0?l=0:o<0?l=i+o>0?i+o:0:l=i<o?i:o;let u;a===-1/0?u=0:a<0?u=i+a>0?i+a:0:u=i<a?i:a;const c=u-l>0?u-l:0,h=new r(c);if(ra(h,c),c===0)return h;const d=rt(n);if(ro(d))throw Be(so);let w=0;for(;l<u;)h[w]=Ce(n[l]),++l,++w;return h}subarray(e,t){Ie(this);const n=ye(this),r=sa(n,Le),i=new Xt(rt(n),Us(n),Te(n)),o=RS(i,e,t),a=new r(rt(o),Us(o),Te(o));return ra(a),a}indexOf(e,...t){Ie(this);const n=ye(this),r=Te(n);let i=ss(t[0]);if(i===1/0)return-1;i<0&&(i+=r,i<0&&(i=0));for(let o=i;o<r;++o)if(ps(n,o)&&Ce(n[o])===e)return o;return-1}lastIndexOf(e,...t){Ie(this);const n=ye(this),r=Te(n);let i=t.length>=1?ss(t[0]):r-1;if(i===-1/0)return-1;i>=0?i=i<r-1?i:r-1:i+=r;for(let o=i;o>=0;--o)if(ps(n,o)&&Ce(n[o])===e)return o;return-1}includes(e,...t){Ie(this);const n=ye(this),r=Te(n);let i=ss(t[0]);if(i===1/0)return!1;i<0&&(i+=r,i<0&&(i=0));const o=pi(e);for(let a=i;a<r;++a){const l=Ce(n[a]);if(o&&pi(l)||l===e)return!0}return!1}join(e){Ie(this);const t=ye(this),n=Kf(t);return wS(n,e)}toLocaleString(...e){Ie(this);const t=ye(this),n=Kf(t);return _S(n,...ta(e))}get[Jc](){if(Xr(this))return"Float16Array"}}Uo(Le,"BYTES_PER_ELEMENT",{value:oh});Uo(Le,cl,{});Dg(Le,th);const fl=Le.prototype;Uo(fl,"BYTES_PER_ELEMENT",{value:oh});Uo(fl,es,{value:fl.values,writable:!0,configurable:!0});Dg(fl,ht);function ek(s,e){return s.channels===e.channels}const ia=8;class oa{autoUpdateOutputBuffer=!0;_label;_device;_outputBuffers={};_pipeline;_bindGroups=[];_needsUpdatePipeline=!0;_needsResizeBuffer=!0;_inputs=[];_outputs=[];_uniforms=[];_uniformBuffers={};_width=10;_height=10;_execWidth;_execHeight;_csCode="";_csMain;_csDefine;_groupOffsets={inputs:0,uniforms:1,outputs:2};constructor(e,t,n){this._label=e,this._device=t,this._csMain=n.csMain,this._csDefine=n.csDefine,this._inputs=n.inputs,this._outputs=n.outputs,this._uniforms=n.uniforms,this.autoUpdateOutputBuffer=n.autoUpdateOutputBuffer??!0,n.uniforms.forEach(r=>{this._uniformBuffers[r.label]=t.createBuffer({label:this._label,size:r.data.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this._device.queue.writeBuffer(this._uniformBuffers[r.label],0,r.data)})}setCSCode({csDefine:e,csMain:t}){this._csDefine=e,this._csMain=t,this._needsUpdatePipeline=!0}setSize(e,t){e=Math.ceil(e),t=Math.ceil(t);const n=e!==this._width||t!==this._height;this._width=e,this._height=t,n&&(this._needsResizeBuffer=!0,this._needsUpdatePipeline=!0)}setExecuteSize(e,t){e=Math.ceil(e),t=Math.ceil(t),this._execWidth=e,this._execHeight=t}setOutputParams(e){this.autoUpdateOutputBuffer&&this._updateOutputBuffers(e),this._needsUpdatePipeline=!0}setOutputBuffers(e){this._outputBuffers=Object.keys(e).reduce((t,n)=>(t[n]={buffer:e[n],params:{channels:4}},t),{})}setUniform(e,t){const n=this._uniformBuffers[e];this._device.queue.writeBuffer(n,0,t)}getOutput(e){return this._needsResizeBuffer&&this.autoUpdateOutputBuffer&&(this._resizeOutputBuffers(),this._needsResizeBuffer=!1),this._outputBuffers[e].buffer}dispose(){Object.keys(this._uniformBuffers).forEach(e=>{this._uniformBuffers[e].destroy()}),Object.keys(this._outputBuffers).forEach(e=>{this._outputBuffers[e].buffer.destroy()})}_createBuffer(e){const t=this._width*this._height*4*4;return this._device.createBuffer({label:this._label,size:Math.max(t,80),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC})}_resizeOutputBuffers(){const e=this._outputBuffers;for(const t in e){const{buffer:n,params:r}=e[t];n.destroy(),e[t].buffer=this._createBuffer(r)}}_updateOutputBuffers(e){const t=this._outputBuffers;for(const n in e){const r=e[n];if(!ek(r,t[n]?.params||{})){t[n]?.buffer.destroy();const i=this._createBuffer(r);t[n]={buffer:i,params:r}}}}_updatePipeline(e,t){if(!this._needsUpdatePipeline)return;this._needsUpdatePipeline=!1;const n=this._device,r=this._getFullCs(e,t);r!==this._csCode&&(this._csCode=r,this._pipeline=n.createComputePipeline({label:this._label,layout:"auto",compute:{module:n.createShaderModule({label:this._label,code:r}),entryPoint:"main"}}),this._updateBindGroups())}_getFullCs(e,t){const n=this._inputs,r=this._uniforms;let i=0;const o=this._groupOffsets={inputs:0,uniforms:0,outputs:0};return n.length>0&&i++,r.length>0&&(o.uniforms=i,i++),o.outputs=i,`
${n.sort().map((l,u)=>{const c=`@group(${o.inputs}) @binding(${u}) `,h=`in_${l}`;return t[l]==="texture"?`${c} var ${h}: texture_2d<f32>;`:`${c} var<storage, read> ${h}: array<vec${e[l].channels}f>;`}).join(`
`)}
${this._uniforms.map((l,u)=>`@group(${o.uniforms}) @binding(${u}) var<uniform> ${l.label}: ${l.type};`).join(`
`)}

${this._outputs.map((l,u)=>`@group(${o.outputs}) @binding(${u}) var<storage, read_write> out_${l}: array<vec${this._outputBuffers[l].params.channels}f>;`).join(`
`)}
${this._csDefine??""}
@compute @workgroup_size(${ia}, ${ia}, 1)
fn main(@builtin(global_invocation_id) globalId: vec3u) {
${this._csMain}
}
`}_updateBindGroups(){const e=[],t=this._device,n=this._groupOffsets;this._uniforms.length>0&&(e[n.uniforms]=t.createBindGroup({label:this._label,layout:this._pipeline.getBindGroupLayout(n.uniforms),entries:this._uniforms.map((r,i)=>({binding:i,resource:{buffer:this._uniformBuffers[r.label]}}))})),this._bindGroups=e}createPass(e,t){this._needsResizeBuffer&&this.autoUpdateOutputBuffer&&(this._resizeOutputBuffers(),this._needsResizeBuffer=!1);const n=this._inputs.reduce((o,a)=>(o[a]=t[a].buffer?"buffer":"texture",o),{});this._updatePipeline(t,n);const r=this._groupOffsets;this._inputs.length>0&&(this._bindGroups[r.inputs]=this._device.createBindGroup({label:this._label,layout:this._pipeline.getBindGroupLayout(r.inputs),entries:this._inputs.map((o,a)=>({binding:a,resource:t[o].buffer?{buffer:t[o].buffer}:t[o].texture.createView()}))})),this._bindGroups[r.outputs]=this._device.createBindGroup({label:this._label,layout:this._pipeline.getBindGroupLayout(r.outputs),entries:this._outputs.map((o,a)=>({binding:a,resource:{buffer:this._outputBuffers[o].buffer}}))});const i=e.beginComputePass();i.setPipeline(this._pipeline),this._bindGroups.forEach((o,a)=>{i.setBindGroup(a,o)}),i.dispatchWorkgroups(Math.ceil((this._execWidth??this._width)/ia),Math.ceil((this._execHeight??this._height)/ia),1),i.end()}}const ah=1412.83765,lh=1.64593172,uh=.431384981,ch=-.00294139609,hh=.192653254,fh=.00626026094,dh=.998620152,Qg=15794576e-13,Jg=.0322087631,ey=.00223151711,ty=.370974749;function ny(s){return s<=Qg?s=ah*s:s<=Jg?s=lh*Math.pow(s,uh)+ch:s=hh*Math.log(s+fh)+dh,s}function tk(s){return s<=ey?s=s/ah:s<=ty?s=Math.pow((s-ch)/lh,1/uh):s=Math.exp((s-dh)/hh)-fh,s}const nk=65504,sy=ny(nk),ry=1/sy,iy=sy;class mu{x;y;width;height;constructor(e,t,n,r){this.x=e,this.y=t,this.width=n,this.height=r}}function sk({data:s,channels:e}){let t=0;for(let o=0;o<s.length;o+=e){const a=s[o],l=s[o+1],u=s[o+2],c=.212671*a+.71516*l+.072169*u;t+=Math.log2(c+1e-4)}const n=s.length/e,r=t/n;return .18/Math.pow(2,r)}function rk({data:s,channels:e,inputScale:t}){const n=new Float32Array(s.length);n.set(s);for(let r=0;r<n.length;r+=e)for(let i=0;i<3;i++){let o=n[r+i]*t;n[r+i]=ny(o)*ry}return n}function ik({data:s,channels:e,inputScale:t}){const n=new Float32Array(s.length);n.set(s);const r=1/t;for(let i=0;i<n.length;i+=e)for(let o=0;o<3;o++){let a=n[i+o]*iy;n[i+o]=tk(a)*r}return n}const Xf=`
const a = ${ah};
const b = ${lh};
const c = ${uh};
const d = ${ch};
const e = ${hh};
const f = ${fh};
const g = ${dh};
const y0 =${Qg};
const y1 =${Jg};
const x0 =${ey};
const x1 =${ty};

const normScale = ${ry};
const rcpNormScale = ${iy};
`;class ok{_device;_isHDR;_inputPassAux;_inputPassColor;_outputPass;_copyPass;_isInputTexture;constructor(e,t){this._device=e,this._isHDR=t;const n=[{label:"inputScale",type:"f32",data:new Float32Array([1])},{label:"inputSize",type:"vec2<f32>",data:new Float32Array(2)},{label:"outputSize",type:"vec2<f32>",data:new Float32Array(2)},{label:"inputOffset",type:"vec2<f32>",data:new Float32Array(2)}];this._inputPassAux=new oa("inputPassAux",this._device,{inputs:["color","albedo","normal"],outputs:["color","albedo","normal"],uniforms:n,csDefine:"",csMain:""}),this._inputPassColor=new oa("inputPassColor",this._device,{inputs:["color"],outputs:["color"],uniforms:n,csDefine:"",csMain:""}),this._outputPass=new oa("outputPass",this._device,{inputs:["color","raw"],outputs:["color"],uniforms:[{label:"inputScale",type:"f32",data:new Float32Array([1])},{label:"inputSize",type:"vec2<f32>",data:new Float32Array(2)},{label:"outputSize",type:"vec2<f32>",data:new Float32Array(2)},{label:"imageSize",type:"vec2<f32>",data:new Float32Array(2)},{label:"inputOffset",type:"vec2<f32>",data:new Float32Array(2)},{label:"outputOffset",type:"vec2<f32>",data:new Float32Array(2)}],csDefine:"",csMain:""}),this._copyPass=new oa("copyPass",this._device,{inputs:["color"],outputs:["color"],autoUpdateOutputBuffer:!1,uniforms:[{label:"size",type:"vec2<f32>",data:new Float32Array(2)}],csMain:`
let outIdx = i32(globalId.x + globalId.y * u32(size.x));
out_color[outIdx] = textureLoad(in_color, globalId.xy, 0);
`}),this._inputPassAux.setOutputParams({color:{channels:3},albedo:{channels:3},normal:{channels:3}}),this._inputPassColor.setOutputParams({color:{channels:3}}),this._outputPass.setOutputParams({color:{channels:4}})}_updatePasses(e,t=!1){if(this._isInputTexture!=null&&this._isInputTexture===e)return;this._isInputTexture=e;const n=this._isHDR,r=`
${Xf}
fn PUForward(y: f32) -> f32 {
  if (y <= y0) {
    return a * y;
  } else if (y <= y1) {
    return b * pow(y, c) + d;
  } else {
    return e * log(y + f) + g;
  }
}`;function i(a){return e?`textureLoad(in_${a}, globalId.xy + vec2u(inputOffset), 0)`:`in_${a}[inIdx]`}const o=`
let x = f32(globalId.x);
let y = f32(globalId.y);
let inIdx = i32((y + inputOffset.y) * inputSize.x + (x + inputOffset.x));
let col = ${i("color")};

let outIdx = i32(y * outputSize.x + x);

if (${t}) {
  // Denoise the inversed alpha. Or the anti aliased edge will be too dark after denoised
  out_color[outIdx] = vec3f(1.0 - col.a);
}
else if (${n}) {
  out_color[outIdx] = vec3f(PUForward(col.r * inputScale), PUForward(col.g * inputScale), PUForward(col.b * inputScale)) * normScale;
}
else {
  out_color[outIdx] = col.rgb;
}
`;this._inputPassAux.setCSCode({csDefine:r,csMain:`
${o}
let alb = ${i("albedo")};
let nor = ${i("normal")};
out_normal[outIdx] = nor.rgb;
out_albedo[outIdx] = alb.rgb;
  `}),this._inputPassColor.setCSCode({csDefine:r,csMain:`
${o}
`}),this._outputPass.setCSCode({csDefine:`
${Xf}
fn PUInverse(y: f32) -> f32 {
  if (y <= x0) {
    return y / a;
  } else if (y <= x1) {
    return pow((y - d) / b, 1 / c);
  } else {
    return exp((y - g) / e) - f;
  }
}
`,csMain:`
let x = f32(globalId.x);
let y = f32(globalId.y);
if (x >= outputSize.x || y >= outputSize.y) {
  return;
}
let inIdx = i32((y + inputOffset.y) * inputSize.x + x + inputOffset.x);
let outIdx = i32((y + outputOffset.y) * imageSize.x + x + outputOffset.x);
let col = in_color[inIdx];
let raw = ${e?"textureLoad(in_raw, globalId.xy + vec2u(outputOffset), 0)":"in_raw[outIdx]"};

if (${t}) {
  out_color[outIdx] = vec4f(raw.rgb, 1.0 - col.r);
}
else if (${n}) {
  out_color[outIdx] = vec4f(
    vec3f(PUInverse(col.r * rcpNormScale), PUInverse(col.g * rcpNormScale), PUInverse(col.b * rcpNormScale)) / inputScale,
    // Pick the alpha
    raw.a
  );
}
else {
  out_color[outIdx] = vec4f(col.rgb, raw.a);
}
`})}setImageSize(e,t){this._inputPassAux.setUniform("inputSize",new Float32Array([e,t])),this._inputPassColor.setUniform("inputSize",new Float32Array([e,t])),this._outputPass.setUniform("imageSize",new Float32Array([e,t])),this._outputPass.setSize(e,t),this._copyPass.setSize(e,t),this._copyPass.setUniform("size",new Float32Array([e,t]))}setInputTile(e){const t=new Float32Array([e.width,e.height]);[this._inputPassAux,this._inputPassColor].forEach(n=>{n.setUniform("inputOffset",new Float32Array([e.x,e.y])),n.setUniform("outputSize",t),n.setSize(t[0],t[1])}),this._outputPass.setUniform("inputSize",t)}setOutputTile(e,t){const n=this._outputPass,r=new Float32Array([e.width,e.height]),i=e.x-t.x,o=e.y-t.y;n.setUniform("outputSize",r),n.setUniform("inputOffset",new Float32Array([i,o])),n.setUniform("outputOffset",new Float32Array([e.x,e.y])),n.setExecuteSize(r[0],r[1])}forward(e,t,n,r){const i=e instanceof GPUTexture;this._updatePasses(i,r);const o=this._inputPassAux,a=this._inputPassColor,l=this._device.createCommandEncoder();function u(c){return c instanceof GPUTexture?{texture:c,channels:4}:{buffer:c,channels:4}}return t&&n?o.createPass(l,{color:u(e),albedo:u(t),normal:u(n)}):a.createPass(l,{color:u(e)}),this._device.queue.submit([l.finish()]),t&&n?{color:o.getOutput("color"),albedo:o.getOutput("albedo"),normal:o.getOutput("normal")}:{color:a.getOutput("color")}}inverse(e,t){const r=this._device.createCommandEncoder(),i=this._outputPass;return i.createPass(r,{color:{buffer:e,channels:4},raw:t instanceof GPUBuffer?{buffer:t,channels:4}:{texture:t,channels:4}}),this._device.queue.submit([r.finish()]),i.getOutput("color")}copyInputDataToOutput(e){const t=this._device.createCommandEncoder(),r=this._outputPass.getOutput("color"),i=this._copyPass;e instanceof GPUTexture?(i.setOutputBuffers({color:r}),i.createPass(t,{color:{texture:e,channels:4}})):t.copyBufferToBuffer(e,0,r,0,r.size),this._device.queue.submit([t.finish()])}dispose(){this._outputPass.dispose(),this._inputPassAux.dispose()}}function Yf(s,e){const t=s.buffer;if(e==="Float32")return new Float32Array(s.buffer);const n=new Le(t),r=new Float32Array(n.length);for(let i=0;i<r.length;++i)r[i]=n[i];return r}function ak(s,e){const[t,n,r,i]=e,o=new Float32Array(s.length);for(let a=0;a<t;++a)for(let l=0;l<n;++l)for(let u=0;u<r;++u)for(let c=0;c<i;++c){const h=a*n*r*i+l*r*i+u*i+c,d=u*i*n*t+c*n*t+l*t+a;o[d]=s[h]}return o}function io(s,e){return Math.ceil(s/e)*e}function aa(s){return s.data instanceof GPUBuffer||s.data instanceof GPUTexture}const lk=174,uk=202,oy=16,la=io(lk/2,oy),Zf=io(uk/2,oy);class ck{_hostTensors;_backend;_tfModel;_device;_tileWidth=0;_tileHeight=0;_tileOverlapX=0;_tileOverlapY=0;_aux;_hdr;_dataProcessGPU;_maxTileSize;_tensors=new Map;_modelsCache=new Map;constructor(e,t,n={}){this._hostTensors=e,this._backend=t,this._aux=n.aux||!1,this._hdr=n.hdr||!1,this._maxTileSize=io(n.maxTileSize??512,2),this._device=this._backend.device}getDevice(){return this._device}_buildModel(e){const n=3+(this._aux?6:0),r=this._getTileSizeWithOverlap(),i=this._modelsCache,o=[r.width,r.height].join(",");if(i.has(o)){this._tfModel=i.get(o);return}const a=G3({name:"input",shape:[r.height,r.width,n],dtype:"float32"});this._tfModel=new Fl({inputs:[a],outputs:e?this._addNetLarge(a):this._addNet(a)}),i.set(o,this._tfModel)}_createConv(e,t,n){const r=e+".weight",i=e+".bias",o=this._tensors;let a=o.get(r),l=o.get(i);const u=this._hostTensors.get(r);if(!a){const h=u.desc.dims;a=ma(ak(Yf(u.data,u.desc.dataType),h),[h[2],h[3],h[1],h[0]],"float32"),o.set(r,a)}if(!l){const h=this._hostTensors.get(e+".bias");l=At(Yf(h.data,h.desc.dataType),"float32"),o.set(i,l)}return new Ei({name:e,filters:u.desc.dims[0],kernelSize:u.desc.dims.slice(2,4),useBias:!0,activation:n,padding:"same",weights:[a,l],trainable:!1}).apply(t)}_createConcatConv(e,t,n){const r=new Qc({name:e+"/concat",trainable:!1,axis:3});return this._createConv(e,r.apply([t,n]),"relu")}_createPooling(e){return new Xc({name:e.name+"/pooling",poolSize:[2,2],strides:[2,2],padding:"same",trainable:!1}).apply(e)}_addUpsamplingLayer(e){return new Kc({name:e.name+"/upsampling",size:[2,2],trainable:!1}).apply(e)}_addNet(e){let t=this._createConv("enc_conv0",e,"relu");const n=t=this._createPooling(this._createConv("enc_conv1",t,"relu")),r=t=this._createPooling(this._createConv("enc_conv2",t,"relu")),i=t=this._createPooling(this._createConv("enc_conv3",t,"relu")),o=t=this._createPooling(this._createConv("enc_conv4",t,"relu"));return t=this._createConv("enc_conv5a",o,"relu"),t=this._addUpsamplingLayer(this._createConv("enc_conv5b",t,"relu")),t=this._createConcatConv("dec_conv4a",t,i),t=this._addUpsamplingLayer(this._createConv("dec_conv4b",t,"relu")),t=this._createConcatConv("dec_conv3a",t,r),t=this._addUpsamplingLayer(this._createConv("dec_conv3b",t,"relu")),t=this._createConcatConv("dec_conv2a",t,n),t=this._addUpsamplingLayer(this._createConv("dec_conv2b",t,"relu")),t=this._createConcatConv("dec_conv1a",t,e),t=this._createConv("dec_conv1b",t,"relu"),t=this._createConv("dec_conv0",t,"relu"),t}_addNetLarge(e){let t=this._createConv("enc_conv1a",e,"relu");const n=t=this._createPooling(this._createConv("enc_conv1b",t,"relu"));t=this._createConv("enc_conv2a",t,"relu");const r=t=this._createPooling(this._createConv("enc_conv2b",t,"relu"));t=this._createConv("enc_conv3a",t,"relu");const i=t=this._createPooling(this._createConv("enc_conv3b",t,"relu"));t=this._createConv("enc_conv4a",t,"relu");const o=t=this._createPooling(this._createConv("enc_conv4b",t,"relu"));return t=this._createConv("enc_conv5a",o,"relu"),t=this._addUpsamplingLayer(this._createConv("enc_conv5b",t,"relu")),t=this._createConcatConv("dec_conv4a",t,i),t=this._addUpsamplingLayer(this._createConv("dec_conv4b",t,"relu")),t=this._createConcatConv("dec_conv3a",t,r),t=this._addUpsamplingLayer(this._createConv("dec_conv3b",t,"relu")),t=this._createConcatConv("dec_conv2a",t,n),t=this._addUpsamplingLayer(this._createConv("dec_conv2b",t,"relu")),t=this._createConcatConv("dec_conv1a",t,e),t=this._createConv("dec_conv1b",t,"relu"),t=this._createConv("dec_conv1c",t,"relu"),t}_updateModel(e,t){const n=this._hostTensors.has("enc_conv1b.weight"),r=this._maxTileSize;let i=r,o=r,a=n?Zf:la,l=n?Zf:la;e<r+la*2&&(i=io(e,r/2),e<=r&&(a=0)),t<r+la*2&&(o=io(t,r/2),t<=r&&(l=0));const u=Math.max(i,o),c=Math.max(a,l);i=u,o=u,a=c,l=c,(i!==this._tileWidth||o!==this._tileHeight||a!==this._tileOverlapX||l!==this._tileOverlapY||!this._tfModel)&&(this._tileWidth=i,this._tileHeight=o,this._tileOverlapX=a,this._tileOverlapY=l,this._buildModel(n))}_getTileSizeWithOverlap(){return{width:this._tileWidth+2*this._tileOverlapX,height:this._tileHeight+2*this._tileOverlapY}}_processImageData(e,t,n,r){const i=e.data,o=i.length/4,a=this._aux?9:3,l=new Float32Array(o*a);if(t&&!n||n&&!t)throw new Error("Normal map and albedo map are both required");if(t&&n&&(t.width!==n.width||t.height!==n.height||e.width!==t.width||e.height!==t.height))throw new Error("Image size mismatch");const u=t?.data,c=n?.data;for(let h=0;h<i.length;h+=4){const d=h/4*a;for(let w=0;w<3;w++)r?l[d+w]=i[h+w]:l[d+w]=i[h+w]/255,u&&(l[d+w+3]=u[h+w]/255),c&&(l[d+w+6]=c[h+w]/255)}return l}_readTile(e,t,n,r){const i=new Float32Array(n.width*n.height*t);for(let o=0;o<n.height;o++)for(let a=0;a<n.width;a++){const l=((o+n.y)*r+(a+n.x))*t,u=(o*n.width+a)*t;for(let c=0;c<t;c++)i[u+c]=e[l+c]}return i}_writeTile(e,t,n,r,i,o){const{data:a,width:l}=e,u=n.x-t.x,c=n.y-t.y;for(let h=0;h<n.height;h++)for(let d=0;d<n.width;d++){const w=((h+c)*i+d+u)*3,k=((h+n.y)*l+(d+n.x))*4;for(let A=0;A<3;A++)o?a[k+A]=r[w+A]:a[k+A]=Math.min(Math.max(r[w+A]*255,0),255);e.data[k+3]=o?1:255}}_executeTile(e,t,n,r,i,o,a,l,u){const c=this._aux?9:3,h=this._tileOverlapX,d=this._tileOverlapY;let w=this._getTileSizeWithOverlap(),k={width:this._tileWidth,height:this._tileHeight},A=r>0?r*k.width-h:0,m=Math.min(A+w.width,o);A=Math.max(m-w.width,0);let S=i>0?i*k.height-d:0,b=Math.min(S+w.height,a);S=Math.max(b-w.height,0);const f=w.width,v=w.height,_=new mu(A,S,f,v);let E,D=1;const M=this._device;let $=this._dataProcessGPU;if(e instanceof Float32Array){let I=this._readTile(e,c,_,o);l&&(D=sk({data:I,channels:c}),I=rk({data:I,channels:c,inputScale:D})),E=ma(I,[1,v,f,c],"float32")}else{$||($=this._dataProcessGPU=new ok(M,l)),$.setImageSize(o,a),$.setInputTile(_),r===0&&i===0&&$.copyInputDataToOutput(e.color);const{color:I,albedo:N,normal:L}=$.forward(e.color,this._aux?e.albedo:void 0,this._aux?e.normal:void 0,u),W=X=>{const V=ma({buffer:X,zeroCopy:!0},[1,v,f,4]);return bo(V,[0,0,0,0],[1,v,f,3])};if(this._aux){const X=[I,N,L].map(V=>W(V));E=O1(X,3)}else E=W(I)}let C;const g=this._tfModel.predict(E),p=Math.min(k.width,o),y=Math.min(k.height,a),x=new mu(r*p,i*y,p,y);if(x.width=Math.min(x.width,o-x.x),x.height=Math.min(x.height,a-x.y),e instanceof Float32Array){let I=g.dataSync();l&&(I=ik({data:I,channels:3,inputScale:D})),this._writeTile(n,_,x,I,w.width,l);for(let N=0;N<y;N++)for(let L=0;L<p;L++){const W=(N*p+L)*4,X=((N+x.y)*o+(L+x.x))*4;for(let V=0;V<4;V++)t.data[W+V]=n.data[X+V]}}else{$.setOutputTile(x,_);const I=T1(g,[[0,0],[0,0],[0,0],[0,1]]);C=$.inverse(I.dataToGPU().buffer,e.color)}return C}tileExecute({color:e,albedo:t,normal:n,done:r,progress:i,denoiseAlpha:o}){if(this._aux&&(!t||!n))throw new Error("Normal map and albedo map are both required");if(!this._aux&&(t||n))throw new Error("Normal map and albedo map are not required");const a=e.width,l=e.height;this._updateModel(a,l);const u=this._hdr||!1;let c;aa(e)||(c=this._processImageData(e,t,n,u));const h=this._tileWidth,d=this._tileHeight,w=Math.ceil(l/d),k=Math.ceil(a/h);function A(v,_){return u?{data:new Float32Array(v*_*4),width:v,height:_}:new ImageData(v,_)}const m=aa(e)?void 0:A(a,l),S=aa(e)?void 0:A(Math.min(h,a),Math.min(d,l));let b=!1;const f=(v,_)=>{if(b)return;let E;K.startScope(),E=this._executeTile(aa(e)?{color:e.data,albedo:t?.data,normal:n?.data}:c,S,m,v,_,a,l,u,o),K.endScope();const D=m||{data:E,width:a,height:l};i?.(D,S,new mu(v*h,_*d,h,d),v+_*k,k*w),v+1<k||_+1<w?requestAnimationFrame(()=>{v+1<k?f(v+1,_):_+1<w&&f(0,_+1)}):r(D)};return f(0,0),()=>{b=!0}}dispose(){this._tfModel?.dispose(),this._dataProcessGPU?.dispose(),this._tensors.forEach(e=>e.dispose())}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Jt=ge();Jt.registerFlag("WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE",()=>15);Jt.registerFlag("WEBGPU_CPU_FORWARD",()=>!0);Jt.registerFlag("WEBGPU_MATMUL_PROGRAM_TYPE",()=>-1);Jt.registerFlag("WEBGPU_USE_NAIVE_CONV2D_TRANSPOSE",()=>!0);Jt.registerFlag("WEBGPU_USE_LOW_POWER_GPU",()=>!1);Jt.registerFlag("WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD",()=>1e3);Jt.registerFlag("WEBGPU_USE_PROFILE_TOOL",()=>!1);Jt.registerFlag("WEBGPU_IMPORT_EXTERNAL_TEXTURE",()=>!0);Jt.registerFlag("WEBGPU_USE_NAIVE_CONV2D_DEBUG",()=>!1);Jt.registerFlag("WEBGPU_THRESHOLD_TO_INCREASE_WORKGROUPS_FOR_MATMUL",()=>-1);Jt.registerFlag("WEBGPU_CONV_SEPARATE_IM2COL_SHADER",()=>!1);Jt.registerFlag("WEBGPU_PRINT_SHADER",()=>"");Jt.registerFlag("WEBGPU_ENGINE_COMPILE_ONLY",()=>!1);/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class hk{constructor(e){e&&(this.vendor=e.vendor,this.architecture=e.architecture,this.intelGPUGeneration=this.getIntelGPUGeneration())}getIntelGPUGeneration(){if(this.isIntel()){if(this.architecture.startsWith("gen"))return Number(this.architecture.match(/\d+/));if(this.architecture.startsWith("xe"))return 12}return 0}isIntel(){return this.vendor==="intel"}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class fk{constructor(e){this.device=e,this.numUsedBuffers=0,this.numFreeBuffers=0,this.freeBuffers=new Map,this.usedBuffers=new Map,this.numBytesUsed=0,this.numBytesAllocated=0}acquireBuffer(e,t,n=!1,r=!0){let i;const o=Qf(e,t);return r?(this.freeBuffers.has(o)||this.freeBuffers.set(o,[]),this.freeBuffers.get(o).length>0?(i=this.freeBuffers.get(o).pop(),this.numFreeBuffers--):(i=this.device.createBuffer({size:e,usage:t,mappedAtCreation:n}),this.numBytesAllocated+=e)):(i=this.device.createBuffer({size:e,usage:t,mappedAtCreation:n}),this.numBytesAllocated+=e),this.usedBuffers.has(o)||this.usedBuffers.set(o,[]),this.usedBuffers.get(o).push(i),this.numUsedBuffers++,this.numBytesUsed+=e,i}releaseBuffer(e,t=!0){if(this.freeBuffers.size===0)return;const n=e.size,r=e.usage,i=Qf(n,r),o=this.usedBuffers.get(i),a=o.indexOf(e);if(a<0)throw new Error("Cannot find the buffer in buffer manager");o[a]=o[o.length-1],o.pop(),this.numUsedBuffers--,this.numBytesUsed-=n,t?(this.freeBuffers.get(i).push(e),this.numFreeBuffers++):(e.destroy(),this.numBytesAllocated-=n)}getNumUsedBuffers(){return this.numUsedBuffers}getNumFreeBuffers(){return this.numFreeBuffers}dispose(){this.freeBuffers.forEach((e,t)=>{e.forEach(n=>{n.destroy()})}),this.usedBuffers.forEach((e,t)=>{e.forEach(n=>{n.destroy()})}),this.freeBuffers=new Map,this.usedBuffers=new Map,this.numUsedBuffers=0,this.numFreeBuffers=0,this.numBytesUsed=0,this.numBytesAllocated=0}}function Qf(s,e){return`${s}_${e}`}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class dk{constructor(e){this.device=e,this.numUsedTextures=0,this.numFreeTextures=0,this.freeTextures=new Map,this.usedTextures=new Map,this.numBytesUsed=0,this.numBytesAllocated=0}acquireTexture(e,t,n,r){const i=ed(n),o=e*t*i,a=Jf(e,t,n,r);if(this.freeTextures.has(a)||this.freeTextures.set(a,[]),this.usedTextures.has(a)||this.usedTextures.set(a,[]),this.numBytesUsed+=o,this.numUsedTextures++,this.freeTextures.get(a).length>0){this.numFreeTextures--;const u=this.freeTextures.get(a).shift();return this.usedTextures.get(a).push(u),u}this.numBytesAllocated+=o;const l=this.device.createTexture({size:[e,t],format:n,usage:r});return this.usedTextures.get(a).push(l),l}releaseTexture(e){if(this.freeTextures.size===0)return;const t=e.width,n=e.height,r=e.format,i=e.usage,o=Jf(t,n,r,i);this.freeTextures.has(o)||this.freeTextures.set(o,[]),this.freeTextures.get(o).push(e),this.numFreeTextures++,this.numUsedTextures--;const a=this.usedTextures.get(o),l=a.indexOf(e);if(l<0)throw new Error("Cannot release a texture that was never provided by this texture manager");a.splice(l,1);const u=ed(r),c=t*n*u;this.numBytesUsed-=c}getNumUsedTextures(){return this.numUsedTextures}getNumFreeTextures(){return this.numFreeTextures}dispose(){this.freeTextures.forEach((e,t)=>{e.forEach(n=>{n.destroy()})}),this.usedTextures.forEach((e,t)=>{e.forEach(n=>{n.destroy()})}),this.freeTextures=new Map,this.usedTextures=new Map,this.numUsedTextures=0,this.numFreeTextures=0,this.numBytesUsed=0,this.numBytesAllocated=0}}function Jf(s,e,t,n){return`${s}_${e}_${t}_${n}`}function ed(s){if(s==="rgba8unorm")return 16;throw new Error(`${s} is not supported!`)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pk(s,e){if(Math.max(...s)>5)throw new Error("Cannot symbolically compute strides for rank > 6 tensor.");const t=s.length,n="xyzwuv",r=s.map(o=>`${e}.${n[o]}`),i=new Array(t-1);i[t-2]=r[t-1];for(let o=t-3;o>=0;--o)i[o]=`(${i[o+1]} * ${r[o+1]})`;return i}const mk=(s,e,t)=>`
          {
            var oldValue = 0;
            loop {
              let newValueF32 = bitcast<f32>(oldValue) + (${e});
              let newValue = bitcast<i32>(newValueF32);
              let res = atomicCompareExchangeWeak(${s}, oldValue, newValue);
              if res.exchanged {
                break;
              }
              oldValue = res.old_value;
            }
          }`;/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var dl;(function(s){s[s.FROM_PIXELS=0]="FROM_PIXELS",s[s.DRAW=1]="DRAW"})(dl||(dl={}));const gk=(s,e,t,n,r)=>{const i={dtype:n.dtype,shape:n.shape},o=bk(t,i,e),a=s.createShaderModule({code:o,label:e.constructor.name});let l=ge().get("WEBGPU_PRINT_SHADER");if(l!==""){l=l.toLowerCase();const u=l.split(",");(l==="all"||u.some(c=>e.shaderKey.toLowerCase().includes(c)))&&(console.group(e.shaderKey),console.debug(o),console.groupEnd())}return r?s.createComputePipelineAsync({compute:{module:a,entryPoint:"_start"},label:e.constructor.name,layout:"auto"}):s.createComputePipeline({compute:{module:a,entryPoint:"_start"},label:e.constructor.name,layout:"auto"})},xe=(s,e="f32")=>{switch(s){case 1:return`${e}`;case 2:return`vec2<${e}>`;case 3:return`vec3<${e}>`;case 4:return`vec4<${e}>`;default:throw new Error(`${s}-component ${e} is not supported.`)}};function kt(s){if(s<=1)return"i32";if(s===2)return"vec2<i32>";if(s===3)return"vec3<i32>";if(s===4)return"vec4<i32>";if(s===5)return"vec5";if(s===6)return"vec6";throw Error(`GPU for rank ${s} is not yet supported`)}function wr(s){if(s===0)return"x";if(s===1)return"y";if(s===2)return"z";if(s===3)return"w";if(s===4)return"u";if(s===5)return"v";throw Error(`Index ${s} is not yet supported`)}function st(...s){let e;switch(s.length){case 0:e=`
        fn main()
      `;break;case 1:e=`
        fn main(${s[0]} : i32)
      `;break;default:throw Error("Unreachable")}return e}function td(s,e){let t;return t=`
     ${yk(e)}
      fn _start(@builtin(local_invocation_id) LocalId : vec3<u32>,
                @builtin(global_invocation_id) GlobalId : vec3<u32>,
                @builtin(local_invocation_index) LocalIndex: u32,
                @builtin(workgroup_id) WorkgroupId : vec3<u32>,
                @builtin(num_workgroups) NumWorkgroups : vec3<u32>) {
        localId = LocalId;
        localIndex = LocalIndex;
        globalId = GlobalId;
        numWorkgroups = NumWorkgroups;
        workgroupId = WorkgroupId;
        ${s?"main(getGlobalIndex());":"main();"};
      }
    `,t}function yk(s){return`
  @compute @workgroup_size(${s.workgroupSize[0]}, ${s.workgroupSize[1]}, ${s.workgroupSize[2]})
`}function bk(s,e,t){const n=[],r=t.workgroupSize[0]*t.workgroupSize[1]*t.workgroupSize[2];if(t.outputComponent=t.outputComponent?t.outputComponent:1,n.push(`

      var<private> localId: vec3<u32>;
      var<private> localIndex: u32;
      var<private> globalId: vec3<u32>;
      var<private> numWorkgroups: vec3<u32>;
      var<private> workgroupId: vec3<u32>;

      // Only used when the y/z dimension of workgroup size is 1.
      fn getGlobalIndex() -> i32 {
        ${ay(t)?"  return i32(globalId.x);":`  return i32((workgroupId.z * numWorkgroups.x * numWorkgroups.y +
                workgroupId.y * numWorkgroups.x + workgroupId.x) * ${r}u +
                localIndex);
        `}
      }
    `),t.pixelsOpType!=null){const k=t.pixelsOpType===dl.FROM_PIXELS?`@group(0) @binding(0) var<storage, read_write> result: array<${Fr(e.dtype,t.outputComponent)}>;`:`@group(0) @binding(1) var<storage, read> inBuf : array<${Fr(s[0].dtype,t.outputComponent)}>;`,A=e.shape.length===3?"vec2<i32>":"i32";n.push(`
        struct Uniform {
          outShapeStrides : ${A},
          size            : i32,
          numChannels     : i32,
          alpha           : f32,
        };

        ${k}
        @group(0) @binding(2) var<uniform> uniforms: Uniform;
      `);const m=sd(t);return[nd,n.join(`
`),gu(e.shape),t.getUserCode(),td(m,t)].join(`
`)}let i,o,a="struct Uniforms { NAN : f32, INFINITY : f32, ";t.variableNames.forEach((k,A)=>{const m=kt(s[A].shape.length);a+=`${k.charAt(0).toLowerCase()+k.slice(1)}Shape : ${m}, `,i=s[A].shape.length-1,o=kt(i),a+=`${k.charAt(0).toLowerCase()+k.slice(1)}ShapeStrides: ${o}, `});const l=kt(e.shape.length);a+=`outShape : ${l}, `,i=e.shape.length-1,o=kt(i),a+=`
         outShapeStrides: ${o}, `,t.size&&(a+="size : i32, "),t.uniforms&&(a+=t.uniforms),a+="};",a=Tk(a),n.push(a),t.atomic?n.push(`
      @group(0) @binding(0) var<storage, read_write> result: array<atomic<i32>>;
    `):n.push(`
      @group(0) @binding(0) var<storage, read_write> result: array<${Fr(e.dtype,t.outputComponent)}>;
    `),t.variableNames.forEach((k,A)=>{n.push(`
      @group(0) @binding(${1+A}) var<storage, read> ${k}: array<${t.variableComponents?Fr(s[A].dtype,t.variableComponents[A]):Fr(s[A].dtype,t.outputComponent)}>;
        `)}),a!==""&&n.push(`
      @group(0) @binding(${1+t.variableNames.length}) var<uniform> uniforms: Uniforms;
      `);const u=kk(e.shape,t.dispatchLayout),c=[nd,n.join(`
`)+xk,gu(e.shape),u,Ik(e.shape.length)];t.atomic||c.push(Ek(e.shape,e.dtype,t.outputComponent)),t.variableNames.forEach((k,A)=>{c.push(`${gu(s[A].shape,k)}`)});const h=s.map((k,A)=>Sk(k,e.shape,t.variableComponents?t.variableComponents[A]:t.outputComponent,t.dispatchLayout.x.length===e.shape.length)).join(`
`);c.push(h),c.push(t.getUserCode());const d=sd(t);return c.push(td(d,t)),c.join(`
`)}function wk(s,e,t){let n=s.shaderKey;if(s.pixelsOpType!=null)return n;const r=[],i=[];e.forEach(c=>{r.push(c.shape),i.push(c.dtype)}),r.push(t.shape),i.push(t.dtype);const o=e.map(c=>Va(c.shape,t.shape)),a=e.map(c=>cn(c.shape,t.shape)).join("_"),l=o.map(c=>c.join("_")).join(";"),u=ay(s)?"flatDispatch":"";return n+="_"+(s.workgroupSize?s.workgroupSize.join(","):"")+r.map(c=>c.length).join(",")+i.join(",")+s.variableNames.join(",")+l+a+u,n}const nd=`
  struct vec5 {x: i32, y: i32, z: i32, w: i32, u: i32};
  struct vec6 {x: i32, y: i32, z: i32, w: i32, u: i32, v: i32};

  // Checks whether coordinates lie within the bounds of the shape.
  fn coordsInBounds2D(coord : vec2<i32>, shape : vec2<i32>) -> bool {
    return all(coord >= vec2<i32>(0)) && all(coord < shape);
  }
  fn coordsInBounds3D(coord : vec3<i32>, shape : vec3<i32>) -> bool {
    return all(coord >= vec3<i32>(0)) && all(coord < shape);
  }
  fn coordsInBounds4D(coord : vec4<i32>, shape : vec4<i32>) -> bool {
    return all(coord >= vec4<i32>(0)) && all(coord < shape);
  }

  fn getIndexFromCoords1D(coord : i32, shape : i32) -> i32 {
    return coord;
  }
  fn getIndexFromCoords2D(coords : vec2<i32>, shape : vec2<i32>) -> i32 {
    return dot(coords, vec2<i32>(shape.y, 1));
  }
  fn getIndexFromCoords3D(coords : vec3<i32>, shape : vec3<i32>) -> i32 {
    return dot(coords, vec3<i32>(shape.y * shape.z, shape.z, 1));
  }
  fn getIndexFromCoords4D(coords : vec4<i32>, shape : vec4<i32>) -> i32 {
    return dot(coords, vec4<i32>(
        shape.y * shape.z * shape.w, shape.z * shape.w, shape.w, 1));
  }
  fn getIndexFromCoords5D(coords : vec5, shape : vec5) -> i32 {
    let shapeStrides: vec5 = vec5(shape.y * shape.z * shape.w * shape.u, shape.z * shape.w * shape.u, shape.w * shape.u, shape.u, 1);
    return coords.x*shapeStrides.x + coords.y*shapeStrides.y + coords.z*shapeStrides.z + coords.w*shapeStrides.w + coords.u*shapeStrides.u;
  }
  fn getIndexFromCoords6D(coords : vec6, shape : vec6) -> i32 {
    let shapeStrides: vec6 = vec6(shape.y * shape.z * shape.w * shape.u * shape.v, shape.z * shape.w * shape.u * shape.v, shape.w * shape.u * shape.v, shape.u * shape.v, shape.v, 1);
    return coords.x*shapeStrides.x + coords.y*shapeStrides.y + coords.z*shapeStrides.z + coords.w*shapeStrides.w + coords.u*shapeStrides.u + coords.v*shapeStrides.v;
  }

  // NaN defination in IEEE 754-1985 is :
  //   - sign = either 0 or 1.
  //   - biased exponent = all 1 bits.
  //   - fraction = anything except all 0 bits (since all 0 bits represents infinity).
  // https://en.wikipedia.org/wiki/IEEE_754-1985#Representation_of_non-numbers
  fn isnan(val: f32) -> bool {
    let floatToUint: u32 = bitcast<u32>(val);
    return (floatToUint & 0x7fffffffu) > 0x7f800000u;
  }
  fn isnanVec4(val : vec4<f32>) -> vec4<bool> {
    let floatToUint: vec4<u32> = bitcast<vec4<u32>>(val);
    return (floatToUint & vec4<u32>(0x7fffffffu)) > vec4<u32>(0x7f800000u);
  }
`,xk=`
  fn isinf(val: f32) -> bool {
    return abs(val) == uniforms.INFINITY;
  }
`;function gu(s,e=""){const t=s.length,n=e!==""?`get${e.charAt(0).toUpperCase()+e.slice(1)}CoordsFromIndex`:"getCoordsFromIndex",r=e!==""?`${e.charAt(0).toLowerCase()+e.slice(1)}ShapeStrides`:"outShapeStrides";if(t<=1)return`fn ${n}(index : i32) -> i32 { return index; }`;const i=Zt(s),o=kt(t),a=[];for(let u=0;u<t;u++)a.push(`d${u}`);if(i.length===1)return`    fn ${n}(index : i32) -> vec2<i32> {
      let d0 = index / uniforms.${r}; let d1 = index - d0 * uniforms.${r};
      return vec2<i32>(d0, d1);
    }`;let l;return l="var index2 = index;"+i.map((u,c)=>{const h=`let ${a[c]} = index2 / uniforms.${r}.${wr(c)}`,d=c===i.length-1?`let ${a[c+1]} = index2 - ${a[c]} * uniforms.${r}.${wr(c)}`:`index2 = index2 - ${a[c]} * uniforms.${r}.${wr(c)}`;return`${h}; ${d};`}).join(""),`
    fn ${n}(index : i32) -> ${o} {
      ${l}
      return ${o}(${a.join(",")});
    }
  `}function _k(s,e){const t=s.name,n=s.shape.length,r=kt(n),i="get"+t.charAt(0).toUpperCase()+t.slice(1),o=["d0","d1","d2","d3","d4","d5"].slice(0,n),a=o.map(c=>`${c} : i32`).join(", ");if(n<1)return`
      fn ${i}() -> ${xe(e)} {
        return ${xe(e)}(${t}[0]);
      }
    `;const l=`uniforms.${t.charAt(0).toLowerCase()+t.slice(1)}Shape`;let u=`${n}D`;return n===0&&(u="1D"),`
    fn ${i}(${a}) -> ${xe(e)} {
      return ${xe(e)}(${t}[getIndexFromCoords${u}(${r}(${o.join(",")}),
        ${l})${e===1?"":` / ${e}`}]);
    }
   `}function vk(s,e,t,n){const r=s.name,i=r.charAt(0).toUpperCase()+r.slice(1),o="get"+i+"ByOutput",a=s.shape.length,l=e.length,u=kt(l);if(cn(s.shape,e)&&n)return`
    fn ${o}Index(globalIndex : i32) -> ${xe(t)} {
      return ${xe(t)}(${r}[globalIndex]);
    }

    fn ${o}Coords(coords : ${u}) -> ${xe(t)} {
      return ${xe(t)}(${r}[${l>1?"getOutputIndexFromCoords(coords)":"coords"}${t===1?"":` / ${t}`}]);
    }
    `;const c=Va(s.shape,e),h=l-a;let d="";if(a===0)return`
    fn ${o}Index(globalIndex : i32) -> ${xe(t)}{
      return get${i}();
    }

    fn ${o}Coords(coords : ${u}) -> ${xe(t)}{
      return get${i}();
    }
  `;l<2&&c.length>=1?d="coords = 0;":d=c.map(m=>`coords.${wr(m+h)} = 0;`).join(`
`);let w="";if(l<2&&a>0)w="coords";else if(l>1){const m=kt(a),S=s.shape.map((b,f)=>`coords.${wr(f+h)}`).join(", ");w=`${m}(${S})`}else w="coords";const k=`uniforms.${r.charAt(0).toLowerCase()+r.slice(1)}Shape`,A=`${a}D`;return`
  fn ${o}Index(globalIndex : i32) -> ${xe(t)} {
    var coords = getCoordsFromIndex(globalIndex);
    ${d}
    return ${xe(t)}(${r}[getIndexFromCoords${A}(${w}, ${k})${t===1?"":` / ${t}`}]);
  }

  fn ${o}Coords(coordsIn : ${u}) -> ${xe(t)} {
    var coords = coordsIn;
    ${d}
    return ${xe(t)}(${r}[getIndexFromCoords${A}(${w}, ${k})${t===1?"":` / ${t}`}]);
  }
`}function Sk(s,e,t,n){let r=_k(s,t);return s.shape.length<=e.length&&(r+=vk(s,e,t,n)),r}function kk(s,e){const{x:t,y:n=[],z:r=[]}=e,i=s.length,o=t.length+n.length+r.length;if(o!==i)return"";if(t.length===i)return`fn getOutputCoords() -> ${kt(i)}{
    let globalIndex = getGlobalIndex();
    return getCoordsFromIndex(globalIndex);
  }
  `;let a="";const l=[t,n,r];for(let d=0;d<l.length;d++){const w=l[d];if(w.length!==0)if(w.length===1)a+=`let d${w[0]} = i32(globalId[${d}]);`;else{const k=pk(w,"uniforms.outShape");a+=`var index${d} = i32(globalId[${d}]);`;for(let A=0;A<k.length;A++)a+=`let d${w[A]} = index${d} / ${k[A]};`,A===k.length-1?a+=`let d${w[A+1]} = index${d} - d${w[A]} * ${k[A]};`:a+=`index${d} = index${d} - d${w[A]} * ${k[A]};`}}const u=[];for(let d=0;d<o;d++)u.push(`d${d}`);const c=kt(o);let h=`fn getOutputCoords() -> ${c} {
  ${a}
`;return u.length===0?h+=`return ${c}(0); }`:h+=`return ${c}(${u.join(",")}); }`,h}function Ik(s){let e="";switch(s){case 0:case 1:e+=`
        fn getOutputIndexFromCoords(coords : i32) -> i32 {
          return coords;
        }
        `;break;case 2:e+=`
        fn getOutputIndexFromCoords(coords : vec2<i32>) -> i32 {
          return dot(coords, vec2<i32>(uniforms.outShapeStrides, 1));
        }
        `;break;case 3:e+=`
        fn getOutputIndexFromCoords(coords : vec3<i32>) -> i32 {
          return dot(coords, vec3<i32>(uniforms.outShapeStrides.x, uniforms.outShapeStrides.y, 1));
        }
        `;break;case 4:e+=`
        fn getOutputIndexFromCoords(coords : vec4<i32>) -> i32 {
          return dot(coords, vec4<i32>(
            uniforms.outShapeStrides.x, uniforms.outShapeStrides.y, uniforms.outShapeStrides.z, 1));
        }
        `;break;case 5:e+=`
        fn getOutputIndexFromCoords(coords : vec5) -> i32 {
          return coords.x * uniforms.outShapeStrides.x +
              coords.y * uniforms.outShapeStrides.y +
              coords.z * uniforms.outShapeStrides.z +
              coords.w * uniforms.outShapeStrides.w +
              coords.u;
        }
        `;break;case 6:e+=`
        fn getOutputIndexFromCoords(coords : vec6) -> i32 {
          return coords.x * uniforms.outShapeStrides.x +
              coords.y * uniforms.outShapeStrides.y +
              coords.z * uniforms.outShapeStrides.z +
              coords.w * uniforms.outShapeStrides.w +
              coords.u * uniforms.outShapeStrides.u +
              coords.v;
        }
        `;break;default:R(!1,()=>`Unsupported ${s}D shape`);break}return e}function ay(s){return s.dispatch[1]===1&&s.dispatch[2]===1}function Fr(s,e=1){if(s==="float32")return xe(e,"f32");if(s==="int32"||s==="bool")return xe(e,"i32");throw new Error(`type ${s} is not supported.`)}function Ek(s,e,t){const n=s.length,r=Fr(e,t);let i=`fn setOutputAtIndex(flatIndex : i32, value : ${xe(t)}) {
      result[flatIndex] = ${r}(value);
    }

    fn setOutputAtIndexI32(flatIndex : i32, value : ${xe(t,"i32")}) {
      result[flatIndex] = ${r}(value);
    }
    `;if(n>=2){const o=["d0","d1","d2","d3","d4","d5"].slice(0,n),a=kt(n);i+=`
      fn setOutputAtCoords(${o.map(l=>`${l} : i32`).join(", ")}, value : ${xe(t)}) {
        let flatIndex = getOutputIndexFromCoords(${a}(${o.join(", ")}));
        setOutputAtIndex(flatIndex${t===1?"":` / ${t}`}, value);
      }
      fn setOutputAtCoordsI32(${o.map(l=>`${l} : i32`).join(", ")}, value : ${xe(t,"i32")}) {
        let flatIndex = getOutputIndexFromCoords(${a}(${o.join(", ")}));
        setOutputAtIndexI32(flatIndex${t===1?"":` / ${t}`}, value);
      }
    `}return i}function Tk(s){const e=/(\w+)\s*:\s*vec(5|6)/g;s=s.replace(e,n=>"@align(16) "+n);const t=/vec(5|6)\s*,\s*(\w+)/g;return s=s.replace(t,(n,r,i)=>`vec${r}, @align(16) ${i}`),s}function sd(s){return!(s.dispatchLayout.hasOwnProperty("y")&&s.dispatchLayout.y.length!==0||s.dispatchLayout.hasOwnProperty("z")&&s.dispatchLayout.z.length!==0)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xr=s=>{let e=1;for(let t=0;t<s.length;t++)e*=s[t];return e};function ft(s,e,t=[1,1,1],n=[1,1,1]){const[r,i,o]=[Math.ceil(xr(s.x.map(a=>e[a]))/(t[0]*n[0])),s.y?Math.ceil(xr(s.y.map(a=>e[a]))/(t[1]*n[1])):1,s.z?Math.ceil(xr(s.z.map(a=>e[a]))/(t[2]*n[2])):1];return[r,i,o]}function Ak(s,e,t,n=!1){const r=[8,8,1],i=[4,4,1];return n||(s<=8&&(i[1]=1),e<=16&&t<=16&&(r[0]=4)),{workgroupSize:r,elementsPerThread:i}}function Ck(s,e,t=!1){if(t)return[8,8,1];const n=xr(s.x.map(i=>e[i])),r=xr(s.y.map(i=>e[i]));return n<=4?[4,16,1]:r<=4?[16,4,1]:[16,16,1]}function Nk(s,e,t=!1){if(t)return[4,4,1];const n=xr(s.x.map(i=>e[i])),r=xr(s.y.map(i=>e[i]));return n<=4?[1,2,1]:r<=4?[2,1,1]:[2,2,1]}function fn(s){return{x:s.map((e,t)=>t)}}function rd(s){if(s==="float32"||s==="int32"||s==="bool"||s==="string")return 4;if(s==="complex64")return 8;throw new Error(`Unknown dtype ${s}`)}function ly(){return!!(typeof globalThis<"u"&&globalThis.navigator&&globalThis.navigator.gpu)}var Vn;(function(s){s[s.MatMulReduceProgram=0]="MatMulReduceProgram",s[s.MatMulSplitKProgram=1]="MatMulSplitKProgram",s[s.MatMulSmallOutputSizeProgram=2]="MatMulSmallOutputSizeProgram",s[s.MatMulPackedProgram=3]="MatMulPackedProgram",s[s.MatMulMax=4]="MatMulMax"})(Vn||(Vn={}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $k=ge().getNumber("WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD"),Dk=(s,e)=>{const t=s.limits.maxComputeWorkgroupsPerDimension,n=e.dispatchLayout,r=e.dispatch;if(r.every(o=>o<=t))return r;R(r[0]>t&&n.y===void 0&&n.z===void 0,()=>"Dispatch size exceeds WebGPU limits in Y or Z dimension.");let i=Math.ceil(Math.sqrt(r[0]));return i>t?(i=Math.ceil(Math.cbrt(r[0])),R(i<=t,()=>"Total dispatch size exceeds WebGPU maximum."),[i,i,i]):[i,i,1]};class zo extends jd{nextDataId(){return zo.nextDataId++}constructor(e,t){if(super(),this.commandQueueOwnedIds=new WeakSet,this.dispatchCountInPass=0,this.disposed=!1,this.downloadWaitMs=0,this.tensorDataPendingDisposal=[],this.queryResolveBuffer=null,this.querySet=null,this.querySetCount=2,this.stagingPendingDisposal=[],this.uniformPendingDisposal=[],this.uploadWaitMs=0,this.hasReadSyncWarned=!1,this.hasTimestampQueryWarned=!1,!ly())throw new Error("WebGPU is not supported on this device");this.pipelineCache={},this.device=e,this.queue=e.queue,this.commandEncoder=null,this.computePassEncoder=null,this.adapterInfo=new hk(t),this.supportTimestampQuery=this.device.features.has("timestamp-query"),this.thresholdToIncreaseWorkgroups=this.adapterInfo.intelGPUGeneration>=12?16:8,this.bufferManager=new fk(this.device),this.textureManager=new dk(this.device),this.tensorMap=new h0(this,eu()),ge().getBool("WEBGPU_USE_PROFILE_TOOL")&&(this.dummyCanvas=document.createElement("canvas"),this.dummyCanvas.width=1,this.dummyCanvas.height=1,this.dummyContext=this.dummyCanvas.getContext("webgpu"),this.dummyContext.configure({device:e,format:"bgra8unorm"}),document.body.appendChild(this.dummyCanvas))}floatPrecision(){return 32}disposeData(e,t=!1){if(!this.tensorMap.has(e))return!0;const n=this.tensorMap.get(e);return t?n.refCount=0:n.refCount--,n.refCount>0?!1:(n.complexTensorInfos!=null&&(this.disposeData(n.complexTensorInfos.real.dataId),this.disposeData(n.complexTensorInfos.imag.dataId)),this.commandQueueOwnedIds.has(e)?(this.tensorDataPendingDisposal.push(e),!0):(this.releaseResource(e),this.tensorMap.delete(e),!0))}memory(){return{numBytesInGPU:this.bufferManager.numBytesUsed,numBytesAllocatedInGPU:this.bufferManager.numBytesAllocated,unreliable:!1}}releaseResource(e){const t=this.tensorMap.get(e);if(!(!t||!t.resource)){if(t.external){t.resource=null;return}t.resource instanceof GPUBuffer?this.bufferManager.releaseBuffer(t.resource):t.resource instanceof GPUTexture&&this.textureManager.releaseTexture(t.resource),t.resource=null}}refCount(e){return this.tensorMap.has(e)?this.tensorMap.get(e).refCount:0}incRef(e){const t=this.tensorMap.get(e);t.refCount++}decRef(e){if(this.tensorMap.has(e)){const t=this.tensorMap.get(e);t.refCount--}}write(e,t,n){if(n==="complex64"&&e!=null)throw new Error("Cannot write to a complex64 dtype. Please use tf.complex(real, imag).");const r={id:this.nextDataId()};return this.tensorMap.set(r,{dtype:n,shape:t,values:e,refCount:1}),r}move(e,t,n,r,i){if(r==="complex64")throw new Error("Cannot write to a complex64 dtype. Please use tf.complex(real, imag).");this.tensorMap.set(e,{dtype:r,shape:n,values:t,refCount:i})}submitQueue(){this.queue.submit([this.commandEncoder.finish()]),this.commandEncoder=null,this.dispatchCountInPass=0,this.commandQueueOwnedIds=new WeakSet,this.tensorDataPendingDisposal.forEach(e=>{this.releaseResource(e),this.tensorMap.delete(e)}),this.uniformPendingDisposal.forEach(e=>this.bufferManager.releaseBuffer(e)),this.stagingPendingDisposal.forEach(e=>this.bufferManager.releaseBuffer(e,!1)),this.tensorDataPendingDisposal=[],this.uniformPendingDisposal=[],this.stagingPendingDisposal=[]}ensureCommandEncoderReady(){this.commandEncoder||(this.commandEncoder=this.device.createCommandEncoder())}endComputePassEncoder(){this.computePassEncoder&&(this.computePassEncoder.end(),this.computePassEncoder=null)}async checkCompileCompletionAsync(){let e;try{e=await Promise.all(Object.values(this.pipelineCache))}catch(t){throw new Error(t.message)}Object.keys(this.pipelineCache).map((t,n)=>{this.pipelineCache[t]=e[n]})}async getBufferData(e){if(ge().getBool("WEBGPU_ENGINE_COMPILE_ONLY"))return console.warn("The data may be invalid since WEBGPU_ENGINE_COMPILE_ONLY is true, this can only be called when WEBGPU_ENGINE_COMPILE_ONLY is false"),null;const t=e.size,n=this.bufferManager.acquireBuffer(t,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);this.ensureCommandEncoderReady(),this.endComputePassEncoder(),this.commandEncoder.copyBufferToBuffer(e,0,n,0,t),this.submitQueue(),await n.mapAsync(GPUMapMode.READ);const r=n.getMappedRange().slice(0);return n.unmap(),n!=null&&this.bufferManager.releaseBuffer(n),ge().getBool("WEBGPU_USE_PROFILE_TOOL")&&(R(this.dummyContext!==void 0,()=>"Fail to get context for profiling tool"),this.dummyContext.getCurrentTexture()),r}convertAndCacheOnCPU(e,t){const n=this.tensorMap.get(e);return n.values=t,n.values}readSync(e){const t=this.tensorMap.get(e),{values:n,complexTensorInfos:r}=t;if(n!=null||t.dtype==="string")return n;if(t.dtype==="complex64"){const A=this.readSync(r.real.dataId),m=this.readSync(r.imag.dataId),S=Kl(uf(A,m).buffer,"float32");return this.convertAndCacheOnCPU(e,S),S}this.hasReadSyncWarned||(this.hasReadSyncWarned=!0,console.warn("The performance of synchronously reading data from GPU to CPU is poor on the webgpu backend, please use asynchronous APIs instead."));const i=["opaque","premultiplied"],o=t.resource,a=o.size;R(a%4===0,()=>"Because there is 4 bytes for one pixel, buffer size must be multiple of 4.");const l=a/4,u=new ArrayBuffer(a),c=256,h=256,d=i.map(A=>new OffscreenCanvas(c,h)),w=new OffscreenCanvas(c,h);this.endComputePassEncoder(),d.map((A,m)=>{const S=A.getContext("webgpu");return S.configure({device:this.device,format:"bgra8unorm",usage:GPUTextureUsage.COPY_DST,alphaMode:i[m]}),S.getCurrentTexture()}).map((A,m)=>{const S=c*4,b=(M,$,C)=>{this.ensureCommandEncoderReady(),this.commandEncoder.copyBufferToTexture({buffer:o,bytesPerRow:S,offset:C},{texture:A},{width:M,height:$}),this.submitQueue();const g=w.getContext("2d",{willReadFrequently:!0});g.clearRect(0,0,M,$),g.drawImage(d[m],0,0);const p=g.getImageData(0,0,M,$).data,y=i[m],x=new Uint8ClampedArray(u,C,M*$*4);for(let I=0;I<x.length;I+=4)if(y==="premultiplied")x[I+3]=p[I+3];else{const N=p[I];x[I]=p[I+2],x[I+1]=p[I+1],x[I+2]=N}},f=Math.floor(l/(c*h));let v=c,_=h,E=0;for(let M=0;M<f;M++)b(v,_,E),E+=c*h*4;const D=l%(c*h);_=Math.floor(D/c),_>0&&(b(v,_,E),E+=_*(c*4)),v=D%c,v>0&&b(v,1,E)});const k=Kl(u,t.dtype);return this.convertAndCacheOnCPU(e,k),k}async read(e){if(!this.tensorMap.has(e))throw new Error(`Tensor ${e} was not registered!`);const t=this.tensorMap.get(e),{values:n}=t;if(n!=null)return n;let r;if(t.dtype==="complex64"){const i=await Promise.all([this.read(t.complexTensorInfos.real.dataId),this.read(t.complexTensorInfos.imag.dataId)]),o=i[0],a=i[1];r=uf(o,a)}else{const i=await this.getBufferData(t.resource);r=Kl(i,t.dtype)}return this.convertAndCacheOnCPU(e,r),r}copyBuffer(e){const t=e.size,n=e.usage,r=this.bufferManager.acquireBuffer(t,n);return this.ensureCommandEncoderReady(),this.endComputePassEncoder(),this.commandEncoder.copyBufferToBuffer(e,0,r,0,t),this.submitQueue(),r}createTensorFromGPUData(e,t,n){let r=e.buffer;if(n==="complex64")throw new Error("Cannot write to a complex64 dtype. ");const i={id:this.nextDataId()};this.tensorMap.set(i,{dtype:n,shape:t,values:null,refCount:1,external:e.zeroCopy});const o=this.tensorMap.get(i),a=rd(o.dtype)*me(o.shape);if(e.buffer.size<a)throw new Error(`GPUBuffer size(${e.buffer.size}) is smaller than tensor size(${a})!`);if((e.buffer.usage&(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC))!==(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC))throw new Error("GPUBuffer.usage should include GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC!");return e.zeroCopy!==!0&&(r=this.copyBuffer(r)),o.resource=r,eu().makeTensorFromDataId(i,t,n,this)}readToGPU(e){const t=this.tensorMap.get(e),{values:n,dtype:r,shape:i,resource:o}=t;if(r==="complex64")throw new Error("Does not support reading buffer for complex64 dtype.");if(o==null)throw n!=null?new Error("Data is not on GPU but on CPU."):new Error("There is no data on GPU or CPU.");const a=o,l=a.size,u=a.usage,c=this.bufferManager.acquireBuffer(l,u);this.ensureCommandEncoderReady(),this.endComputePassEncoder(),this.commandEncoder.copyBufferToBuffer(o,0,c,0,l),this.submitQueue();const h=this.makeTensorInfo(i,r),d=eu().makeTensorFromTensorInfo(h),w=this.tensorMap.get(h.dataId);return w.resource=c,{tensorRef:d,buffer:c}}bufferSync(e){const t=this.readSync(e.dataId);if(e.dtype==="string")try{const n=t.map(r=>Fa(r));return ct(e.shape,e.dtype,n)}catch{throw new Error("Failed to decode encoded string bytes into utf-8")}return ct(e.shape,e.dtype,t)}async time(e){!this.supportTimestampQuery&&!this.hasTimestampQueryWarned&&(console.warn("This device doesn't support timestamp-query extension. Start Chrome browser with flag --enable-dawn-features=allow_unsafe_apis to try it again. Otherwise, zero will be shown for the kernel time when profiling mode is enabled."),this.hasTimestampQueryWarned=!0);const t=this.activeTimers,n=[];let r=!1;this.programTimersStack==null?(this.programTimersStack=n,r=!0):this.activeTimers.push(n),this.activeTimers=n,e();const i=_r(this.activeTimers.map(u=>u.query)).filter(u=>u!=null),o=_r(this.activeTimers.map(u=>u.name)).filter(u=>u!=null);this.activeTimers=t,r&&(this.programTimersStack=null);const a={uploadWaitMs:this.uploadWaitMs,downloadWaitMs:this.downloadWaitMs,kernelMs:null,wallMs:null},l=await Promise.all(i);return a.kernelMs=d0(l),a.getExtraProfileInfo=()=>l.map((u,c)=>({name:o[c],ms:u})).map(u=>`${u.name}: ${u.ms}`).join(", "),this.uploadWaitMs=0,this.downloadWaitMs=0,a}makeTensorInfo(e,t,n){return t==="string"&&n!=null&&n.length>0&&Tl(n[0])&&(n=n.map(i=>fr(i))),{dataId:this.write(n,e,t),shape:e,dtype:t}}tensorToBinding(e){if(!e)return null;const n=this.tensorMap.get(e.dataId).resource;return n instanceof GPUBuffer?{buffer:n}:n instanceof GPUTexture?n.createView():n}uploadToGPU(e){const t=this.tensorMap.get(e);if(t.resource!=null)return;const n=rd(t.dtype)*me(t.shape);let r;const i=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;if(t.values){if(r=this.bufferManager.acquireBuffer(n,i,!0),r.mapState==="unmapped"){const o=this.bufferManager.acquireBuffer(n,GPUBufferUsage.MAP_WRITE|GPUBufferUsage.COPY_SRC,!0,!1),a=o.getMappedRange();t.dtype==="int32"||t.dtype==="bool"?new Int32Array(a).set(t.values):new Float32Array(a).set(t.values),o.unmap(),this.ensureCommandEncoderReady(),this.endComputePassEncoder(),this.commandEncoder.copyBufferToBuffer(o,0,r,0,n),this.stagingPendingDisposal.push(o)}else{const o=r.getMappedRange();t.dtype==="int32"||t.dtype==="bool"?new Int32Array(o).set(t.values):new Float32Array(o).set(t.values),r.unmap()}t.values=null}else r=this.bufferManager.acquireBuffer(n,i);t.resource=r}makeUniforms(e){let t=0,n=0;const r=[];let i=1;e.forEach(l=>{l.data.length===0&&(l.data=[1]);let u;switch(l.data.length){case 1:u=4;break;case 2:u=8;break;case 3:u=16;break;case 4:u=16;break;case 5:u=16;break;case 6:u=16;break;default:R(!1,()=>`Unsupported ${l.data.length}D shape`)}(n===5||n===6)&&(u=16),u>i&&(i=u),t=Math.ceil(t/u)*u,n=l.data.length,r.push(t),t+=l.data.length*4}),t=Math.ceil(t/i)*i;const o=new ArrayBuffer(t);e.forEach((l,u)=>{const c=r[u];l.type==="int32"?new Int32Array(o,c,l.data.length).set(l.data):l.type==="uint32"?new Uint32Array(o,c,l.data.length).set(l.data):new Float32Array(o,c,l.data.length).set(l.data)});const a=this.bufferManager.acquireBuffer(t,GPUBufferUsage.COPY_DST|GPUBufferUsage.UNIFORM);return this.queue.writeBuffer(a,0,o,0,t),this.uniformPendingDisposal.push(a),{offset:0,size:t,buffer:a}}runWebGPUProgram(e,t,n,r,i){if(i||(i=this.makeTensorInfo(e.outputShape,n)),me(i.shape)===0)return this.tensorMap.get(i.dataId).values=li(i.dtype,0),i;this.uploadToGPU(i.dataId),e.dispatch=Dk(this.device,e);const o=t.map((l,u)=>{if(l.dtype==="complex64")throw new Error("GPGPUProgram does not support complex64 input. For complex64 dtypes, please separate the program into real and imaginary parts.");return this.uploadToGPU(l.dataId),{dtype:this.tensorMap.get(l.dataId).dtype,shape:l.shape,name:e.variableNames[u]}});e.shaderKey=wk(e,o,i);const a=ge().getBool("WEBGPU_ENGINE_COMPILE_ONLY");return e.shaderKey in this.pipelineCache||(this.pipelineCache[e.shaderKey]=gk(this.device,e,o,i,a)),e.pipeline=this.pipelineCache[e.shaderKey],a||this.recordAndSubmit(e,i,t,r),i}recordAndSubmit(e,t,n,r){if(e.pipeline instanceof Promise)throw new Error("Please call checkCompileCompletionAsync to ensure parallel compilation is done!");let i=[],o=[];const a="int32";if(e.pixelsOpType==null){i.push({type:"float32",data:[NaN]},{type:"float32",data:[1/0]}),o=n.concat(t).map(w=>w.shape);const d="int32";o.map(w=>{i.push({type:d,data:w});const k=Zt(w);i.push({type:d,data:k})})}else{const d=Zt(t.shape);i.push({type:a,data:d})}if(e.size){const d=me(e.outputShape);i.push({type:a,data:[e.outputComponent?d/e.outputComponent:d]})}r&&(i=[...i,...r]);const l=[this.tensorToBinding(t),...n.map(d=>this.tensorToBinding(d)),this.makeUniforms(i)];n.forEach(d=>{this.commandQueueOwnedIds.add(d.dataId)}),this.commandQueueOwnedIds.add(t.dataId);const u=this.device.createBindGroup({layout:e.pipeline.getBindGroupLayout(0),entries:l.map((d,w)=>({binding:w,resource:d}))}),c=this.activeTimers!=null;this.ensureCommandEncoderReady();const h={};c&&this.supportTimestampQuery?(this.endComputePassEncoder(),this.querySet==null&&(this.querySet=this.device.createQuerySet({type:"timestamp",count:this.querySetCount})),h.timestampWrites={querySet:this.querySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1},this.computePassEncoder=this.commandEncoder.beginComputePass(h)):this.computePassEncoder||(this.computePassEncoder=this.commandEncoder.beginComputePass(h)),this.computePassEncoder.setPipeline(e.pipeline),this.computePassEncoder.setBindGroup(0,u),this.computePassEncoder.dispatchWorkgroups(e.dispatch[0],e.dispatch[1],e.dispatch[2]),this.dispatchCountInPass++,(c||ge().get("WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE")<=this.dispatchCountInPass||e.pixelsOpType===dl.DRAW)&&(this.endComputePassEncoder(),c?this.activeTimers.push({name:e.constructor.name,query:this.getQueryTime()}):this.submitQueue())}async getQueryTime(){if(!this.supportTimestampQuery)return 0;this.queryResolveBuffer==null&&(this.queryResolveBuffer=this.bufferManager.acquireBuffer(this.querySetCount*8,GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST|GPUBufferUsage.QUERY_RESOLVE)),this.commandEncoder.resolveQuerySet(this.querySet,0,this.querySetCount,this.queryResolveBuffer,0);const e=this.bufferManager.acquireBuffer(this.querySetCount*8,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);this.commandEncoder.copyBufferToBuffer(this.queryResolveBuffer,0,e,0,this.querySetCount*8),this.submitQueue(),await e.mapAsync(GPUMapMode.READ);const t=new BigUint64Array(e.getMappedRange()),n=Number(t[1]-t[0])/1e6;return e.unmap(),this.bufferManager.releaseBuffer(e),n}shouldExecuteOnCPU(e,t=$k){return ge().getBool("WEBGPU_CPU_FORWARD")&&e.every(n=>this.tensorMap.get(n.dataId).resource==null&&me(n.shape)<t)}numDataIds(){return this.tensorMap.numDataIds()-this.tensorDataPendingDisposal.length}dispose(){this.disposed||(this.querySet!=null&&this.querySet.destroy(),this.bufferManager.dispose(),this.textureManager.dispose(),this.disposed=!0)}}zo.nextDataId=0;/**
 * @license
 * Copyright 2022 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */ly()&&L1("webgpu",async()=>{const s={powerPreference:ge().get("WEBGPU_USE_LOW_POWER_GPU")?"low-power":"high-performance"},e=await navigator.gpu.requestAdapter(s),t={},n=[];e.features.has("timestamp-query")&&n.push("timestamp-query"),e.features.has("bgra8unorm-storage")&&n.push(["bgra8unorm-storage"]),t.requiredFeatures=n;const r=e.limits;t.requiredLimits={maxComputeWorkgroupStorageSize:r.maxComputeWorkgroupStorageSize,maxComputeWorkgroupsPerDimension:r.maxComputeWorkgroupsPerDimension,maxStorageBufferBindingSize:r.maxStorageBufferBindingSize,maxBufferSize:r.maxBufferSize,maxComputeWorkgroupSizeX:r.maxComputeWorkgroupSizeX,maxComputeInvocationsPerWorkgroup:r.maxComputeInvocationsPerWorkgroup};const i=await e.requestDevice(t),o=await e.requestAdapterInfo();return new zo(i,o)},3);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ok{constructor(e,t,n){this.uniforms="",this.variableNames=["x"],this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=t.map((r,i)=>r[0]+e[i]+r[1]),this.dispatchLayout=fn(this.outputShape),this.dispatch=ft(this.dispatchLayout,this.outputShape,this.workgroupSize),this.xShape=e,t.map((r,i)=>{this.uniforms+=` pad${i} : vec2<i32>,`}),this.offset=n==="reflect"?0:1,this.shaderKey=`mirrorPad_${n}`}getUserCode(){const e=this.xShape.length,t=this.xShape.map((u,c)=>`uniforms.pad${c}[0]`).join(","),n=this.xShape.map((u,c)=>`uniforms.pad${c}[0] + uniforms.xShape${e>1?`[${c}]`:""}`).join(","),r=e===1?"start":"start[i]",i=e===1?"end":"end[i]",o=e===1?"outC":"outC[i]",a=kt(e),l=e>1?["coords[0]","coords[1]","coords[2]","coords[3]"].slice(0,e):"coords";return`
      ${st("index")} {
        if (index < uniforms.size) {
          let start = ${a}(${t});
          let end = ${a}(${n});
          var outC = getCoordsFromIndex(index);
          for (var i = 0; i < ${e}; i = i + 1) {
            if (${o} < ${r}) {
              ${o} = ${r} * 2 - ${o} - ${this.offset};
            } else if(${o} >= ${i}) {
              ${o} = (${i} - 1) * 2 - ${o} + ${this.offset};
            }
          }
          let coords = outC - start;
          setOutputAtIndex(index, getX(${l}));
        }
      }
    `}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Mk={kernelName:yb,backendName:"webgpu",kernelFunc:({inputs:s,attrs:e,backend:t})=>{const{x:n}=s,{paddings:r,mode:i}=e,o=t,a=r.map(c=>({type:"int32",data:[c[0],c[1]]})),l=new Ok(n.shape,r,i);return o.runWebGPUProgram(l,[n],n.dtype,a)}};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ls(s){const{inputs:e}=s,{x:t}=e;return s.backend.incRef(t.dataId),{dataId:t.dataId,shape:t.shape,dtype:t.dtype}}const Pk={kernelName:wc,backendName:"webgpu",kernelFunc:Ls};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rk(s,e=!1){const t=s.length,n=kt(t),r=s.map((h,d)=>`uniforms.pad${d}[0]`).join(","),i=s.map((h,d)=>`uniforms.pad${d}[0] + uniforms.xShape${t>1?`[${d}]`:""}`).join(","),o=t>1?`${n}(${r})`:`${r}`,a=t>1?`${n}(${i})`:`${i}`,l=t>1?"any(paddedCoords < start)":"paddedCoords < start",u=t>1?"any(paddedCoords >= end)":"paddedCoords >= end",c=t>1?["coords[0]","coords[1]","coords[2]","coords[3]"].slice(0,t):"coords";return`
        let start = ${o};
        let end = ${a};
        if (${l} || ${u}) {
          setOutputAtIndex(index, ${e?0:"uniforms.constantValue"});
        } else {
          let coords = paddedCoords - start;
          setOutputAtIndex(index, getX(${c}));
        }
  `}class Lk{constructor(e,t){this.variableNames=["x"],this.uniforms="constantValue : f32,",this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=t.map((n,r)=>n[0]+e[r]+n[1]),this.dispatchLayout=fn(this.outputShape),this.dispatch=ft(this.dispatchLayout,this.outputShape,this.workgroupSize),t.map((n,r)=>{this.uniforms+=` pad${r} : vec2<i32>,`}),this.xShape=e,this.shaderKey="pad"}getUserCode(){return`
      ${st("index")} {
        if (index < uniforms.size) {
          let paddedCoords = getCoordsFromIndex(index);
          ${Rk(this.xShape)}
        }
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Bk{constructor(e){this.variableNames=[],this.outputShape=[],this.uniforms="value : f32,",this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=fn(this.outputShape),this.dispatch=ft(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey="fill"}getUserCode(){return`
    ${st("index")} {
      if (index < uniforms.size) {
        setOutputAtIndex(index, uniforms.value);
      }
    }
  `}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uy(s){const{backend:e,attrs:t}=s,{shape:n,value:r}=t;let{dtype:i}=t;if(i=i||No(r),i==="string"){const o=nt(i,me(n));return o.fill(r),e.makeTensorInfo(n,i,o)}else{const o=new Bk(n),a=[{type:"float32",data:[r]}];return e.runWebGPUProgram(o,[],i,a)}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Fk=s=>{const{inputs:e,backend:t,attrs:n}=s,{x:r}=e,{paddings:i,constantValue:o}=n;if(i.every(u=>cn(u,[0,0])))return Ls({inputs:{x:r},backend:t});if(me(r.shape)===0){const u=i.map((c,h)=>c[0]+r.shape[h]+c[1]);return uy({backend:t,attrs:{shape:u,value:o,dtype:r.dtype}})}const a=[{type:"float32",data:[o]}];i.map(u=>a.push({type:"int32",data:[u[0],u[1]]}));const l=new Lk(r.shape,i);return t.runWebGPUProgram(l,[r],r.dtype,a)},Uk={kernelName:sp,backendName:"webgpu",kernelFunc:Fk};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zk(s){const e=new Float32Array(s.length);for(let t=0;t<s.length;++t)e[t]=Math.abs(s[t]);return e}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dt(s){return(e,t,n,r,i)=>{const o=Ft(e,t),a=o.length,l=Zt(o),u=me(o),c=li(i,u),h=e.length,d=t.length,w=Zt(e),k=Zt(t),A=Va(e,o),m=Va(t,o);if(A.length+m.length===0)for(let S=0;S<c.length;++S)c[S]=s(n[S%n.length],r[S%r.length]);else for(let S=0;S<c.length;++S){const b=gc(S,a,l),f=b.slice(-h);A.forEach(D=>f[D]=0);const v=Tu(f,h,w),_=b.slice(-d);m.forEach(D=>_[D]=0);const E=Tu(_,d,k);c[S]=s(n[v],r[E])}return[c,o]}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wk(s,e,t,n){if(n==="int32"){const r=Int32Array.from(s);return[e,"int32",r]}if(n==="bool"){const r=Cl([0],t),[i,o]=Dt((a,l)=>a!==l?1:0)(e,[],s,r,"bool");return[o,"bool",i]}throw new Error(`Error in Cast: failed to cast ${t} to ${n}`)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Gk=Dt((s,e)=>s+e);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vk(s,e,t,n,r){const i=me(n),o=Rs(r,t);for(let a=0;a<s.length;a++){const l=s[a];if(l<0)throw new Error("Input x must be non-negative!");l>=r||(i>0?o[l]+=e[a]:o[l]+=1)}return o}function qk(s,e,t,n=!1){const r=s.shape[0],i=s.shape[1],o=ct([r,t],e.dtype);for(let a=0;a<r;a++)for(let l=0;l<i;l++){const u=s.get(a,l);if(u<0)throw new Error("Input x must be non-negative!");u>=t||(n?o.set(1,a,u):e.size>0?o.set(o.get(a,u)+e.get(a,l),a,u):o.set(o.get(a,u)+1,a,u))}return o}/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Hk=Dt((s,e)=>s&e);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ws(s){return(e,t,n)=>{const r=nt(t,e.length);for(let i=0;i<e.length;++i)r[i]=s(e[i],n);return r}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const jk=ws(s=>Math.ceil(s));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kk(s,e,t,n){const r=nt(t,me(e));if(n&&t!=="string"){let i=0;s.forEach(o=>{const a=me(o.shape);r.set(o.vals,i),i+=a})}else{let i=0;s.forEach(o=>{const a=t==="string"?Qp(o.vals):o.vals;let l=0;for(let u=0;u<o.shape[0];++u){const c=u*e[1]+i;for(let h=0;h<o.shape[1];++h)r[c+h]=a[l++]}i+=o.shape[1]})}return r}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xk=Dt((s,e)=>s===e?1:0);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Yk=ws(s=>Math.exp(s));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Zk=ws(s=>Math.expm1(s));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qk=ws(s=>Math.floor(s));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Jk=Dt((s,e)=>Math.floor(s/e));/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eI(s,e,t,n,r,i,o,a,l){const u=ct([n,i],t);for(let c=0;c<n;c++){const h=[];let d=0;for(let w=0;w<r;w++){const k=s[c*r+w];d+=k*o[w],h.push(k)}if(d<0||d>=l/i)throw new Error(`Invalid indices: ${h} does not index into ${a}`);for(let w=0;w<i;w++)u.values[c*i+w]=e.get(...e.indexToLoc(d*i+w))}return u}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tI(s,e,t){const n=ct(t,s.dtype);for(let r=0;r<n.size;++r){const o=n.indexToLoc(r).slice(),a=o[0],l=o[2],u=e.locToIndex([a,l]);o[2]=e.values[u];const c=s.locToIndex(o);0<=c&&c<s.values.length&&(n.values[r]=s.values[c])}return n}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const nI=Dt((s,e)=>s>e?1:0);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const sI=Dt((s,e)=>s>=e?1:0);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const rI=Dt((s,e)=>s<e?1:0);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const iI=Dt((s,e)=>s<=e?1:0);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function oI(s,e,t){const n=(e-s)/(t-1),r=Rs(t,"float32");r[0]=s;for(let i=1;i<r.length;i++)r[i]=r[i-1]+n;return r}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const aI=ws(s=>Math.log(s));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lI(s,e,t,n){const r=li(n,me(t));for(let i=0;i<r.length;++i){const o=i*e;let a=s[o];for(let l=0;l<e;++l){const u=s[o+l];(Number.isNaN(u)||u>a)&&(a=u)}r[i]=a}return r}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const uI=Dt((s,e)=>Math.max(s,e));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cI=Dt((s,e)=>Math.min(s,e));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const cy=Dt((s,e)=>s*e);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hI(s,e,t){const n=a1(-1,t);return cy([],e,n,s,t)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const fI=Dt((s,e)=>s!==e?1:0);/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dI(s,e,t,n,r){const i=e.length,o=me(e),a=Zt(e),l=Zt(r),u=li(t,me(r));for(let c=0;c<o;++c){const h=gc(c,i,a),d=new Array(h.length);for(let k=0;k<d.length;k++)d[k]=h[n[k]];const w=Tu(d,i,l);u[w]=s[c]}return u}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pI(s,e,t,n){const[r,i]=Tc(s,n),o=xc(e,"int32"),a=Rs(me(r),o),l=me(i);for(let u=0;u<a.length;++u){const c=u*l;let h=1;for(let d=0;d<l;++d)h*=t[c+d];a[u]=h}return{outVals:a,outShape:r,outDtype:o}}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mI(s,e,t){s.forEach((n,r)=>{if(n<0||n>=t){const i=gc(r,e.length,Zt(e)).join(",");throw new Error(`indices[${i}] = ${n} is not in [0, ${t})`)}})}function gI(s,e){for(let t=0;t<s.length;++t){const n=s[t],r=t===s.length-1?e:s[t+1].length;if(n.length===0)throw new Error("Ragged splits may not be empty");if(n[0]<0)throw new Error("Ragged splits must be non-negative");if(n[n.length-1]>r)throw new Error("Ragged splits must not point past values");for(let i=1;i<n.length;++i)if(n[i-1]>n[i])throw new Error("Ragged splits must be sorted in ascending order")}}function yI(s,e,t,n){const r=[];let i=0;const o=e.length-1+t.length,a=new Array(o).fill(null).map(()=>[0]);gI(t,n);let l=1;for(let u=0;u<e.length-1;++u){l*=e[u];const c=e[u+1];for(let h=1;h<l+1;++h)a[u].push(h*c)}for(let u=0;u<s.length;++u){let c=s[u],h=s[u]+1;for(let d=0;d<t.length;++d){const w=t[d],k=d+e.length-1;if(k>=0){const A=a[k],m=A[A.length-1]-w[c];for(let S=c;S<h;++S)a[k].push(w[S+1]+m)}c=w[c],h=w[h]}h!==c&&(r.push([c,h]),i+=h-c)}return{outSplits:a,valueSlices:r,numValues:i}}function bI(s){const e=[];for(let t=0;t<s.length;++t){const n=s[t].length,r=nt("int32",n);e.push(r),s[t].forEach((i,o)=>r[o]=i)}return e}function id(s,e){const t=s.slice(0,e);for(;t.length<e;)t.push(1);for(let n=e;n<s.length;n++)t[e-1]*=s[n];return t}function wI(s,e,t,n,r,i){const o=id(e,2)[1],a=id(i,2)[1];let l=0;for(const u of t)for(let c=u[0];c<u[1];++c){for(let h=0;h<n;++h)r[l*a+h]=s[c*o+h];++l}}function xI(s,e,t,n,r){const i=e.slice();i[0]=r;const o=nt(t,me(i)),a=s.length,l=a===0?0:a/e[0];return wI(s,e,n,l,o,i),[o,i]}function _I(s,e,t,n,r,i,o,a){if(s.length===0)throw new Error("paramsNestedSplits must be non empty");if(e[0].length===0)throw new Error("Split tensors must not be scalars");const l=e[0][0]-1;if(mI(i,o,l),n.length===0)throw new Error("params.rank must be nonzero");const u=n[0],{outSplits:c,valueSlices:h,numValues:d}=yI(i,o,s,u),w=bI(c),k=xI(t,n,r,h,d);return[w,k[0],k[1]]}/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const od=2147483647;function vI(s,e,t,n,r,i,o){if(e.length>1)throw new Error("starts must be a scalar or vector");if(r.length>1)throw new Error("limits must be a scalar or vector");if(o.length>1)throw new Error("deltas must be a scalar or vector");const a=e.length===0,l=r.length===0,u=o.length===0,c=[];a||c.push(e[0]),l||c.push(r[0]),u||c.push(o[0]);for(let m=1;m<c.length;++m)if(c[m]!==c[m-1])throw new Error("starts, limits, and deltas must have the same shape");const h=c.length===0?1:c[0],d=nt("int32",h+1);d[0]=0;for(let m=0;m<h;++m){const S=a?s[0]:s[m],b=l?n[0]:n[m],f=u?i[0]:i[m];if(f===0)throw new Error("Requires delta != 0");let v;if(f>0&&b<S||f<0&&b>S)v=0;else if(v=Math.ceil(Math.abs((b-S)/f)),v>od)throw new Error(`Requires ((limit - start) / delta) <= ${od}`);d[m+1]=d[m]+v}const w=d[h],k=nt(t,w);let A=0;for(let m=0;m<h;++m){const S=d[m+1]-d[m];let b=a?s[0]:s[m];const f=u?i[0]:i[m];for(let v=0;v<S;++v)k[A++]=b,b+=f}return[d,k]}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var tn=Kn;class pl{constructor(e,t,n,r,i,o,a,l,u,c){this.shape=e,this.shapeShape=t,this.values=n,this.valuesShape=r,this.valuesDType=i,this.defaultValue=o,this.defaultValueShape=a,this.rowPartitionValues=l,this.rowPartitionValuesShapes=u,this.rowPartitionTypes=sv(c),this.raggedRank=rv(this.rowPartitionTypes)}getRowPartitionTypeByDimension(e){return this.rowPartitionTypes[0]===tn.FIRST_DIM_SIZE?this.rowPartitionTypes[e+1]:this.rowPartitionTypes[e]}getRowPartitionTensor(e){return this.rowPartitionTypes[0]===tn.FIRST_DIM_SIZE?this.rowPartitionValues[e+1]:this.rowPartitionValues[e]}getMaxWidth(e){const t=this.getRowPartitionTensor(e-1);switch(this.getRowPartitionTypeByDimension(e-1)){case tn.VALUE_ROWIDS:return pl.getMaxWidthValueRowID(t);case tn.ROW_SPLITS:return pl.getMaxWidthRowSplit(t);default:throw new Error(`Cannot handle partition type ${tn[this.getRowPartitionTypeByDimension(e-1)]}`)}}static getMaxWidthRowSplit(e){const t=e.length;if(t===0||t===1)return 0;let n=0;for(let r=0;r<t-1;++r){const i=e[r+1]-e[r];i>n&&(n=i)}return n}static getMaxWidthValueRowID(e){const t=e.length;if(t===0)return 0;let n=0,r=e[0],i=0;for(let o=1;o<t;++o){const a=e[o];a!==r&&(r=a,i=Math.max(o-n,i),n=o)}return Math.max(t-n,i)}tensorShapeFromTensor(e,t,n=!0){if(t.length===0){if(e[0]===-1)return[];throw new Error("The only valid scalar shape tensor is the fully unknown shape specified as -1.")}return ld(e,n)}calculateOutputSize(e){const t=this.valuesShape,n=this.defaultValueShape;iv(n,t);const r=this.tensorShapeFromTensor(this.shape,this.shapeShape),o=nv(this.raggedRank,r,t);o[0]<0&&(o[0]=e);for(let a=1;a<=this.raggedRank;++a)o[a]<0&&(o[a]=this.getMaxWidth(a));return o}calculateFirstParentOutputIndex(e,t,n){const r=Math.min(e,n),i=[];let o=0;for(let a=0;a<r;++a,o+=t)i.push(o);for(let a=r;a<e;++a)i.push(-1);return R(i.length===e,()=>"Final length of result must be equal to firstDimension."),i}calculateOutputIndexRowSplit(e,t,n,r){const i=e.length,o=[];for(let a=0;a<i-1;++a){const l=e[a+1]-e[a];let u=Math.min(r,l),c=t[a];c===-1&&(u=0);for(let h=0;h<u;++h)o.push(c),c+=n;for(let h=0;h<l-u;++h)o.push(-1)}if(i>0&&o.length!==e[i-1])throw new Error("Invalid row split size.");return o}calculateOutputIndexValueRowID(e,t,n,r){const i=e.length,o=[];if(i===0)return[];let a=0,l=e[0];if(l>=t.length)throw new Error(`Got currentValueRowId=${l}, which is not less than ${t.length}`);let u=t[l];o.push(u);for(let c=1;c<i;++c){const h=e[c];if(h===l)u>=0&&(++a,a<r?u+=n:u=-1);else{if(a=0,l=h,h>=t.length)throw new Error(`Got nextValueRowId=${h} which is not less than ${t.length}`);u=t[h]}o.push(u)}if(o.length!==e.length)throw new Error("Invalid row ids.");return o}calculateOutputIndex(e,t,n,r){const i=this.getRowPartitionTensor(e),o=this.getRowPartitionTypeByDimension(e);switch(o){case tn.VALUE_ROWIDS:return this.calculateOutputIndexValueRowID(i,t,n,r);case tn.ROW_SPLITS:if(i.length-1>t.length)throw new Error(`Row partition size is greater than output size: ${i.length-1} > ${t.length}`);return this.calculateOutputIndexRowSplit(i,t,n,r);default:throw new Error(`Unsupported partition type: ${tn[o]}`)}}getFirstDimensionSize(){const e=this.rowPartitionValues[0];if(this.rowPartitionTypes.length===0)throw new Error("No row_partition_types given.");const t=this.rowPartitionTypes[0];switch(t){case tn.FIRST_DIM_SIZE:return e[0];case tn.VALUE_ROWIDS:throw new Error("Cannot handle VALUE_ROWIDS in first dimension.");case tn.ROW_SPLITS:return this.rowPartitionValuesShapes[0][0]-1;default:throw new Error(`Cannot handle type ${tn[t]}`)}}compute(){if(this.rowPartitionValues[0].length<=0)throw new Error("Invalid first partition input. Tensor requires at least one element.");const t=this.getFirstDimensionSize(),n=this.calculateOutputSize(t),r=new Array(this.raggedRank+1);r[r.length-1]=1;for(let l=r.length-2;l>=0;--l)r[l]=r[l+1]*n[l+1];const i=ld(n,!1),o=nt(this.valuesDType,me(i));if(r[0]*n[0]>0){let l=this.calculateFirstParentOutputIndex(t,r[0],n[0]);for(let u=1;u<=this.raggedRank;++u)l=this.calculateOutputIndex(u-1,l,r[u],n[u]);this.setOutput(this.raggedRank,l,o,i)}return[i,o]}setOutput(e,t,n,r){if(n.length===0)return;const i=this.values,o=n;let a=r.slice();a=a.slice(e+1);const l=me(a),u=t.length;let c=this.defaultValue;if(c.length!==l&&c.length!==1){const k=this.defaultValueShape;Q(()=>{const A=ae(c,k);c=ga(A,a).dataSync()})}let h=0,d=0,w=0;for(let k=0;k<=u;++k){let A=k<u?t[k]:-1;if(A===w){++w;continue}if(d<w){const m=i.subarray(h*l),S=o.subarray(d*l),b=(w-d)*l;ad(S,m,b)}if(k>=u){const m=n.length;A=Math.floor(m/l)}if(A>w)if(this.defaultValue.length===1)o.subarray(w*l,A*l).fill(this.defaultValue[0]),w=A;else for(;A>w;){const m=o.slice(w*l);ad(m,c,l),++w}A<0?(h=k+1,d=w):(h=k,d=w,w=d+1)}}}function ad(s,e,t){for(let n=0;n<t;n++)s[n]=e[n]}function ld(s,e){const t=[];for(let n of s){if(n<0){if(!e)throw new Error(`Dimension ${n} must be >= 0`);if(n<-1)throw new Error(`Dimension ${n} must be >= -1`);n=-1}t.push(n)}return t}function SI(s,e,t,n,r,i,o,a,l,u){return new pl(s,e,t,n,r,i,o,a,l,u).compute()}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kI(s,e,t,n){const r=s===e,i=s<e&&t<0,o=e<s&&t>1;if(r||i||o)return Rs(0,n);const a=Math.abs(Math.ceil((e-s)/t)),l=Rs(a,n);e<s&&t===1&&(t=-1),l[0]=s;for(let u=1;u<l.length;u++)l[u]=l[u-1]+t;return l}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const II=ws(s=>1/Math.sqrt(s));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function EI(s,e,t,n,r,i,o,a,l,u){const c=[n/r,r],h=s.values,d=e.values;if(n===0)return ct(t,e.dtype);const w=l instanceof Ua?l:ct(c,e.dtype);typeof l=="string"||typeof l=="number"?w.values.fill(l):typeof l=="boolean"&&w.values.fill(+l);for(let k=0;k<i;k++){const A=[];let m=0;for(let S=0;S<o;S++){const b=h[k*o+S];A.push(b),m+=b*a[S]}if(m<0||m>=n/r)throw new Error(`Invalid indices: ${A} does not index into ${t}`);for(let S=0;S<r;S++)u?w.values[m*r+S]+=d[k*r+S]:w.values[m*r+S]=e.rank===0?d[0]:d[k*r+S]}return w}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const TI=ws(s=>1/(1+Math.exp(-s)));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function AI(s,e,t,n,r){const i=X_(n,e,t),o=me(t),a=Zt(n);if(i){const h=Y_(e,a);return r==="string"?s.slice(h,h+o):s.subarray(h,h+o)}const l=r==="string"?Qp(s):s,u=ct(n,r,l),c=ct(t,r);for(let h=0;h<c.size;++h){const d=c.indexToLoc(h),w=d.map((k,A)=>k+e[A]);c.set(u.get(...w),...d)}return r==="string"?Iv(c.values):c.values}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function CI(s,e,t,n,r,i,o){const a=e[0],l=i[0],u=new Array(l),c=new Array(a),h=e[1];if(l===0){if(a!==0)throw new Error(pv(a));const m=nt(t,0),S=nt(r,0);return[m,[0,h],S,u,c]}let d=!0,w=0;const k=new Array(l).fill(0);for(let m=0;m<a;++m){const S=s[m*h];if(S<0)throw new Error(mv(m,S));if(S>=l)throw new Error(gv(m,S,l));++k[S],d=d&&S>=w,w=S}let A=!0;for(let m=0;m<l;++m){const S=k[m]===0;u[m]=S,A=A&&!S,k[m]=Math.max(k[m],1),m>0&&(k[m]+=k[m-1])}if(A&&d){const m=s,S=n;for(let b=0;b<a;++b)c[b]=b;return[m,[a,h],S,u,c]}else{const m=k[l-1],S=nt(t,m*h),b=nt(r,m),f=new Array(l).fill(0);for(let v=0;v<a;++v){const _=s[v*h],E=f[_],D=(_===0?0:k[_-1])+E;f[_]++;for(let M=0;M<h;++M)S[D*h+M]=s[v*h+M];b[D]=n[v],c[v]=D}for(let v=0;v<l;++v)if(f[v]===0){const E=v===0?0:k[v-1];S[E*h+0]=v;for(let D=1;D<h;++D)S[E*h+D]=0;b[E]=o}return[S,[m,h],b,u,c]}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function NI(s,e,t,n,r){const i=me(n),o=e[0],a=r.length,l=[];let u=1,c=-1;for(let m=0;m<a;++m){const S=r[m];if(S===-1){if(c!==-1)throw new Error(yv(c,m));c=m,l.push(1)}else{if(S<0)throw new Error(bv(m,S));u*=S,l.push(S)}}if(c!==-1){if(u<=0)throw new Error(wv());const m=Math.trunc(i/u);if(u*m!==i)throw new Error(xv(n,l));l[c]=m}if(me(l)!==i)throw new Error(_v(n,l));const d=n.length,w=[];if(d>0){w[d-1]=1;for(let m=d-2;m>=0;--m)w[m]=w[m+1]*n[m+1]}const k=[];if(a>0){k[a-1]=1;for(let m=a-2;m>=0;--m)k[m]=k[m+1]*l[m+1]}const A=nt(t,o*a);for(let m=0;m<o;++m){let S=0;for(let b=0;b<d;++b)S+=s[m*d+b]*w[b];for(let b=0;b<a;++b)A[m*a+b]=Math.trunc(S/k[b]),S%=k[b]}return[A,[o,a],l]}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $I(s,e,t,n,r,i=!1,o=0){const a=n.length,l=[e[0],s.length/e[0]],u=l[1],h=a>0?r[a-1]+1:0;if(h<0)throw new Error(cf());const d=e.slice();d[0]=h;const w=d.reduce((f,v)=>f*v,1),k=nt(t,w);if(a===0)return h>0&&k.fill(o),[k,d];if(h<=0)throw new Error(cf());let A=0,m=1,S=0,b=r[A];for(;;){let f=0;if(m<a){if(f=r[m],b===f){++m;continue}if(b>=f)throw new Error(vv())}if(b<0||b>=h)throw new Error(Sv(b,h));b>S&&k.fill(o,S*u,b*u);for(let v=A;v<m;++v){const _=n[v];if(_<0||_>=l[0])throw new Error(kv(v,n[v],l[0]));for(let E=0;E<u;E++)k[b*u+E]+=s[_*u+E]}if(i)for(let v=0;v<u;v++)k[b*u+v]/=m-A;if(A=m,++m,S=b+1,b=f,m>a)break}return S<h&&k.fill(o,S*u,h*u),[k,d]}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const DI=ws(s=>Math.sqrt(s));/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const OI=Dt((s,e)=>{const t=s-e;return t*t});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const MI=ws((s,e)=>{const{pattern:t,replaceGlobal:n,rewrite:r}=e;return s.replace(new RegExp(t,n?"g":""),r)});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function PI(s,e,t,n){const r=ct(s,e.dtype);for(let i=0;i<r.size;i++){const o=r.indexToLoc(i),a=new Array(o.length);for(let l=0;l<a.length;l++)a[l]=o[l]*t[l]+n[l];r.set(e.get(...a),...o)}return r}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class RI{constructor(e,t,n,r,i,o){this.separator=fr(e),this.nGramWidths=t,this.leftPad=fr(n),this.rightPad=fr(r),this.padWidth=i,this.preserveShort=o}getPadWidth(e){return Math.min(this.padWidth<0?e-1:this.padWidth,e-1)}getNumNGrams(e,t){const n=this.getPadWidth(t);return Math.max(0,e+2*n-t+1)}createNGrams(e,t,n,r,i,o){for(let a=0;a<i;++a){const l=this.getPadWidth(o),u=Math.max(0,l-a),c=Math.max(0,l-(i-(a+1))),h=o-(u+c),d=t+(u>0?0:a-l);let w=0;w+=u*this.leftPad.length;for(let b=0;b<h;++b)w+=e[d+b].length;w+=c*this.rightPad.length;const k=u+c+h-1;w+=k*this.separator.length,n[r+a]=new Uint8Array(w);const A=n[r+a];let m=0;const S=b=>b.forEach(f=>A[m++]=f);for(let b=0;b<u;++b)S(this.leftPad),S(this.separator);for(let b=0;b<h-1;++b)S(e[d+b]),S(this.separator);if(h>0){S(e[d+h-1]);for(let b=0;b<c;++b)S(this.separator),S(this.rightPad)}else{for(let b=0;b<c-1;++b)S(this.rightPad),S(this.separator);S(this.rightPad)}}}compute(e,t){const n=e.length,r=t.length;if(r>0){let l=t[0];if(l!==0)throw new Error(`First split value must be 0, got ${l}`);for(let u=1;u<r;++u){let c=t[u]>=l;if(c=c&&t[u]<=n,!c)throw new Error(`Invalid split value ${t[u]}, must be in [${l}, ${n}]`);l=t[u]}if(l!==n)throw new Error(`Last split value must be data size. Expected ${n}, got ${l}`)}const i=r-1,o=nt("int32",r);if(n===0||r===0){const l=new Array(n);for(let u=0;u<=i;++u)o[u]=0;return[l,o]}o[0]=0;for(let l=1;l<=i;++l){const u=t[l]-t[l-1];let c=0;this.nGramWidths.forEach(h=>{c+=this.getNumNGrams(u,h)}),this.preserveShort&&u>0&&c===0&&(c=1),o[l]=o[l-1]+c}const a=new Array(o[i]);for(let l=0;l<i;++l){const u=t[l];let c=o[l];if(this.nGramWidths.forEach(h=>{const d=t[l+1]-t[l],w=this.getNumNGrams(d,h);this.createNGrams(e,u,a,c,w,h),c+=w}),this.preserveShort&&c===o[l]){const h=t[l+1]-t[l];if(h===0)continue;const d=h+2*this.padWidth;this.createNGrams(e,u,a,c,1,d)}}return[a,o]}}function LI(s,e,t,n,r,i,o,a){return new RI(t,n,r,i,o,a).compute(s,e)}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function BI(s,e,t,n){if(!s.length)return;if(e.length===0){for(let i=0;i<s.length;++i)n.push(s.subarray(i,i+1));return}if(e.length===1){const i=e[0];let o=s.indexOf(i);for(;o!==-1;){const a=s.subarray(0,o);(!t||a.length!==0)&&n.push(a),s=s.subarray(o+1),o=s.indexOf(i)}(!t||s.length!==0)&&n.push(s);return}let r=0;for(let i=0;i<s.length+1;i++)if(i===s.length||e.indexOf(s[i])!==-1){const o=s.subarray(r,i);(!t||o.length!==0)&&n.push(o),r=i+1}}function FI(s,e,t){const n=s.length,r=[];let i=0,o=0;const a=new Array(n);for(let d=0;d<n;++d){const w=r.length;BI(s[d],e,t,r);const k=r.length-w;a[d]=k,i+=k,o=Math.max(o,k)}const l=nt("int32",i*2),u=new Array(i),c=[n,o];let h=0;for(let d=0;d<n;++d)for(let w=0;w<a[d];++w)l[h*2]=d,l[h*2+1]=w,u[h]=r[h],++h;return[l,u,c]}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function UI(s,e){const t=nt("int32",s.length);for(let n=0;n<s.length;++n)t[n]=o1(s[n]).modulo(e).getLowBitsUnsigned();return t}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zI=Dt((s,e)=>s-e);/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function WI(s,e){const t=new Array(s.rank);for(let r=0;r<t.length;r++)t[r]=s.shape[r]*e[r];const n=ct(t,s.dtype);for(let r=0;r<n.values.length;++r){const i=n.indexToLoc(r),o=new Array(s.rank);for(let l=0;l<o.length;l++)o[l]=i[l]%s.shape[l];const a=s.locToIndex(o);n.values[r]=s.values[a]}return n}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Wi=(s,e)=>{const t=e.value-s.value;return t===0?s.index-e.index:t};function hy(s,e,t=0,n=s.length-1){for(;n>t;){if(n-t>600){const a=n-t+1,l=e-t+1,u=Math.log(a),c=.5*Math.exp(2*u/3),h=.5*Math.sqrt(u*c*(a-c)/a)*Math.sign(l-a/2),d=Math.max(t,Math.floor(e-l*c/a+h)),w=Math.min(n,Math.floor(e+(a-l)*c/a+h));hy(s,e,d,w)}const r=s[e];let i=t,o=n;for(Pr(s,t,e),Wi(s[n],r)>0&&Pr(s,t,n);i<o;){for(Pr(s,i,o),i++,o--;Wi(s[i],r)<0;)i=i+1;for(;Wi(s[o],r)>0;)o=o-1}Wi(s[t],r)===0?Pr(s,t,o):(o=o+1,Pr(s,o,n)),o<=e&&(t=o+1),e<=o&&(n=o-1)}}function GI(s,e,t,n,r){const i=e[e.length-1],[o,a]=[s.length/i,i],l=li(t,o*n),u=li("int32",o*n);for(let h=0;h<o;h++){const d=h*a,w=s.subarray(d,d+a);let k=new Array(w.length);w.forEach((b,f)=>k[f]={value:b,index:f}),n<k.length&&(hy(k,n),k=k.slice(0,n)),r&&k.sort(Wi);const A=h*n,m=l.subarray(A,A+n),S=u.subarray(A,A+n);for(let b=0;b<n;b++)m[b]=k[b].value,S[b]=k[b].index}const c=e.slice();return c[c.length-1]=n,[ct(c,t,l),ct(c,"int32",u)]}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function VI(s,e,t,n){const r=Co(e,t)[0],i=[1,t[0],1];for(let k=0;k<r;k++)i[0]*=t[k];i[1]=t[r];for(let k=r+1;k<t.length;k++)i[2]*=t[k];const o=new Map,a=new Int32Array(t[r]),l=new Ua(i,n,s),u=[],c=i[0]===1&&i[2]===1;for(let k=0;k<t[r];k++){let A;if(c)A=s[k].toString();else{const S=[];for(let b=0;b<i[0];b++)for(let f=0;f<i[2];f++)S.push(l.get(b,k,f));A=S.join(",")}const m=o.get(A);if(m!=null)a[k]=m;else{const S=o.size;o.set(A,S),a[k]=S,u.push(k)}}const h=i.slice();h[1]=o.size;const d=new Ua(h,n);u.forEach((k,A)=>{for(let m=0;m<i[0];m++)for(let S=0;S<i[2];S++)d.set(l.get(m,k,S),m,A,S)});const w=t.slice();return w[r]=h[1],{outputValues:d.values,outputShape:w,indices:a}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var qI=Object.freeze({__proto__:null,addImpl:Gk,bincountImpl:Vk,bincountReduceImpl:qk,bitwiseAndImpl:Hk,castImpl:Wk,ceilImpl:jk,concatImpl:Kk,equalImpl:Xk,expImpl:Yk,expm1Impl:Zk,floorDivImpl:Jk,floorImpl:Qk,gatherNdImpl:eI,gatherV2Impl:tI,greaterEqualImpl:sI,greaterImpl:nI,lessEqualImpl:iI,lessImpl:rI,linSpaceImpl:oI,logImpl:aI,maxImpl:lI,maximumImpl:uI,minimumImpl:cI,multiplyImpl:cy,negImpl:hI,notEqualImpl:fI,prodImpl:pI,raggedGatherImpl:_I,raggedRangeImpl:vI,raggedTensorToTensorImpl:SI,rangeImpl:kI,rsqrtImpl:II,scatterImpl:EI,sigmoidImpl:TI,simpleAbsImpl:zk,sliceImpl:AI,sparseFillEmptyRowsImpl:CI,sparseReshapeImpl:NI,sparseSegmentReductionImpl:$I,sqrtImpl:DI,squaredDifferenceImpl:OI,staticRegexReplaceImpl:MI,stridedSliceImpl:PI,stringNGramsImpl:LI,stringSplitImpl:FI,stringToHashBucketFastImpl:UI,subImpl:zI,tileImpl:WI,topKImpl:GI,transposeImpl:dI,uniqueImpl:VI});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const{concatImpl:HI,maxImpl:jI,prodImpl:KI,sliceImpl:XI,transposeImpl:YI}=qI;/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class ZI{constructor(e,t){this.variableNames=["source"],this.workPerThread=1,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=t,this.rank=t.length,this.dispatchLayout=fn(this.outputShape),this.dispatch=ft(this.dispatchLayout,this.outputShape,this.workgroupSize,[this.workPerThread,1,1]),this.start=e,this.uniforms=`start : ${kt(e.length)}, `,this.shaderKey="slice"}getUserCode(){const e=kt(this.rank),t=QI(this.rank);let n;return this.start.length===1?n=this.outputShape.map((i,o)=>"sourceLoc = uniforms.start + coords;"):n=this.outputShape.map((i,o)=>`sourceLoc.${ec[o]} = uniforms.start.${wr(o)} + coords.${ec[o]};`),`
      ${st("index")} {
        if (index < uniforms.size) {
          var sourceLoc : ${e};
          let coords = getCoordsFromIndex(index);
          ${n.join(`
`)}
          setOutputAtIndex(index, getSource(${t}));
        }
      }
    `}}const ec=["x","y","z","w","u","v"];function QI(s){if(s===1)return"sourceLoc";if(s<=6)return ec.slice(0,s).map(e=>`sourceLoc.${e}`).join(",");throw Error(`Slicing for rank ${s} is not yet supported`)}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function JI(s){const{inputs:e,backend:t,attrs:n}=s,{x:r}=e,{begin:i,size:o}=n,[a,l]=Z_(r,i,o);if(K_(r,a,l),t.shouldExecuteOnCPU([r])||r.dtype==="string"){const h=t.tensorMap.get(r.dataId),d=XI(h.values,a,l,r.shape,r.dtype);return t.makeTensorInfo(l,r.dtype,d)}if(me(l)===0)return t.makeTensorInfo(l,r.dtype,[]);const u=new ZI(a,l),c=[{type:"int32",data:a}];return t.runWebGPUProgram(u,[r],r.dtype,c)}const eE={kernelName:ip,backendName:"webgpu",kernelFunc:JI};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Oe;(function(s){s[s.ADD=0]="ADD",s[s.ATAN2=1]="ATAN2",s[s.COMPLEX_MULTIPLY_IMAG=2]="COMPLEX_MULTIPLY_IMAG",s[s.COMPLEX_MULTIPLY_REAL=3]="COMPLEX_MULTIPLY_REAL",s[s.DIV=4]="DIV",s[s.ELU_DER=5]="ELU_DER",s[s.EQUAL=6]="EQUAL",s[s.FLOOR_DIV=7]="FLOOR_DIV",s[s.GREATER=8]="GREATER",s[s.GREATER_EQUAL=9]="GREATER_EQUAL",s[s.LESS=10]="LESS",s[s.LESS_EQUAL=11]="LESS_EQUAL",s[s.LOGICAL_AND=12]="LOGICAL_AND",s[s.LOGICAL_OR=13]="LOGICAL_OR",s[s.MAX=14]="MAX",s[s.MIN=15]="MIN",s[s.MOD=16]="MOD",s[s.MUL=17]="MUL",s[s.NOT_EQUAL=18]="NOT_EQUAL",s[s.POW=19]="POW",s[s.PRELU=20]="PRELU",s[s.SQUARED_DIFFERENCE=21]="SQUARED_DIFFERENCE",s[s.SUB=22]="SUB"})(Oe||(Oe={}));const tE="let resultTemp = a + b;",nE="let resultTemp = atan2(a, b);",sE="let resultTemp = areal * breal - aimag * bimag;",rE="let resultTemp = areal * bimag + aimag * breal;",iE="let resultTemp = a / b;",oE="let resultTemp = select(a * (b + 1.0), a, b >= b - b);",aE=`
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a == b);
`,lE=`
  let remainder =
      select(a % b, round(a % b), (round(a) == a) & (round(b) == b));
  let quotient = (a - remainder) / b;
  let resultTemp =
      round(select(quotient, quotient - 1, sign(remainder) == -sign(b)));
`,uE=`
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a > b);
`,cE=`
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a >= b);
`,hE=`
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a < b);
`,fE=`
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a <= b);
`,dE="return f32(a >= 1.0 && b >= 1.0);",pE=`return (vec4<f32>(a >= vec4<f32>(1.0)) *
  vec4<f32>(b >= vec4<f32>(1.0)));`,mE="return f32(a >= 1.0 || b >= 1.0);",gE=`return min(vec4<f32>(a >= vec4<f32>(1.0)) +
  vec4<f32>(b >= vec4<f32>(1.0)), vec4<f32>(1.0));`,yE="let resultTemp = max(a, b);",bE="let resultTemp = min(a, b);",wE=`
  let isNaN = b == 0.;
  var resultTemp = a % b;
  resultTemp = select((resultTemp + b) % b, resultTemp,
      (a < 0. && b < 0.) || (a >= 0. && b > 0.));
`,xE=`
  let isNaN = !vec4<bool>(b);
  var resultTemp = vec4<f32>(a % b);
  if (!((a[0] < 0. && b[0] < 0.) || (a[0] >= 0. && b[0] > 0.))) {
    resultTemp[0] = (resultTemp[0] + b[0]) % b[0];
  }
  if (!((a[1] < 0. && b[1] < 0.) || (a[1] >= 0. && b[1] > 0.))) {
    resultTemp[1] = (resultTemp[1] + b[1]) % b[1];
  }
  if (!((a[2] < 0. && b[2] < 0.) || (a[2] >= 0. && b[2] > 0.))) {
    resultTemp[2] = (resultTemp[2] + b[2]) % b[2];
  }
  if (!((a[3] < 0. && b[3] < 0.) || (a[3] >= 0. && b[3] > 0.))) {
    resultTemp[3] = (resultTemp[3] + b[3]) % b[3];
  }
`,_E="let resultTemp = a * b;",vE=`
  var resultTemp = f32(a != b);
  let valueForNaN = 1.0;
`,SE=`
  var resultTemp = vec4<f32>(a != b);
  let valueForNaN = 1.0;
`,kE=`
  let isNaN = a < 0.0 && floor(b) < b;
  if (b == 0.0) {
    return 1.0;
  }
  var resultTemp = select(sign(a) * pow(abs(a), b), pow(abs(a), b),
      round(abs(b) % 2.0) != 1.0);
`,IE=`
  let isModRound1Bool = vec4<i32>(round(abs(b) % vec4<f32>(2.0))) == vec4<i32>(1);
  let isModRound1 = vec4<f32>(isModRound1Bool);
  let multiplier = sign(a) * isModRound1 + (vec4<f32>(1.0) - isModRound1);
  var resultTemp = multiplier * pow(abs(a), b);

  // Ensure that a^0 = 1, including 0^0 = 1 as this correspond to TF and JS
  let isExpZero = b == vec4<f32>(0.0);
  if (isExpZero.r) {
    resultTemp.r = 1.0;
  }
  if (isExpZero.g) {
    resultTemp.g = 1.0;
  }
  if (isExpZero.b) {
    resultTemp.b = 1.0;
  }
  if (isExpZero.a) {
    resultTemp.a = 1.0;
  }
  let isNaN = (a < vec4<f32>(0.0)) & (floor(b) < b);
`,EE="if (a < 0.0) { return b * a; }  return a;",TE=`
  let aLessThanZero = vec4<f32>(a < vec4<f32>(0.0));
  return (aLessThanZero * (b * a)) + ((vec4<f32>(1.0) - aLessThanZero) * a);
`,AE="let resultTemp = (a - b) * (a - b);",CE="let resultTemp = a - b;";function NE(s,e){let t;do{switch(s){case Oe.ATAN2:t=nE;break;case Oe.MAX:t=yE;break;case Oe.MIN:t=bE;break;case Oe.MOD:t=e?xE:wE;break;case Oe.NOT_EQUAL:t=e?SE:vE;break;case Oe.POW:t=e?IE:kE;break;default:continue}let n,r,i;return e?(n="isnanVec4",r="vec4<f32>",i="vec4<bool>"):(n="isnan",r="f32",i="bool"),`
      let aIsNaN = ${n}(a);
      let aPostLegalization = select(a, ${r}(42), aIsNaN);
      let bIsNaN = ${n}(b);
      let bPostLegalization = select(b, ${r}(42), bIsNaN);
      let isNaN = false;
      let valueForNaN = uniforms.NAN;
      {
        let a = aPostLegalization;
        let b = bPostLegalization;
        ${t}
        return select(
            resultTemp, ${r}(valueForNaN),
            ${i}(isNaN) | aIsNaN | bIsNaN);
      }
    `}while(!1);switch(s){case Oe.ADD:t=tE;break;case Oe.COMPLEX_MULTIPLY_IMAG:t=rE;break;case Oe.COMPLEX_MULTIPLY_REAL:t=sE;break;case Oe.DIV:t=iE;break;case Oe.ELU_DER:t=oE;break;case Oe.EQUAL:t=aE;break;case Oe.FLOOR_DIV:t=lE;break;case Oe.GREATER:t=uE;break;case Oe.GREATER_EQUAL:t=cE;break;case Oe.LESS:t=hE;break;case Oe.LESS_EQUAL:t=fE;break;case Oe.LOGICAL_AND:return e?pE:dE;case Oe.LOGICAL_OR:return e?gE:mE;case Oe.MUL:t=_E;break;case Oe.PRELU:return e?TE:EE;case Oe.SQUARED_DIFFERENCE:t=AE;break;case Oe.SUB:t=CE;break}return`
    ${t}
    return resultTemp;
  `}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var pe;(function(s){s[s.ABS=0]="ABS",s[s.ACOS=1]="ACOS",s[s.ACOSH=2]="ACOSH",s[s.ASIN=3]="ASIN",s[s.ASINH=4]="ASINH",s[s.ATAN=5]="ATAN",s[s.ATANH=6]="ATANH",s[s.CEIL=7]="CEIL",s[s.COS=8]="COS",s[s.COSH=9]="COSH",s[s.ELU=10]="ELU",s[s.ERF=11]="ERF",s[s.EXP=12]="EXP",s[s.EXPM1=13]="EXPM1",s[s.FLOOR=14]="FLOOR",s[s.IS_FINITE=15]="IS_FINITE",s[s.IS_INF=16]="IS_INF",s[s.IS_NAN=17]="IS_NAN",s[s.LINEAR=18]="LINEAR",s[s.LOG=19]="LOG",s[s.LOG1P=20]="LOG1P",s[s.LOGICAL_NOT=21]="LOGICAL_NOT",s[s.NEG=22]="NEG",s[s.RELU=23]="RELU",s[s.RELU6=24]="RELU6",s[s.LEAKYRELU=25]="LEAKYRELU",s[s.RECIPROCAL=26]="RECIPROCAL",s[s.ROUND=27]="ROUND",s[s.RSQRT=28]="RSQRT",s[s.SELU=29]="SELU",s[s.SIGMOID=30]="SIGMOID",s[s.SIGN=31]="SIGN",s[s.SIN=32]="SIN",s[s.SINH=33]="SINH",s[s.SOFTPLUS=34]="SOFTPLUS",s[s.SQRT=35]="SQRT",s[s.SQUARE=36]="SQUARE",s[s.STEP=37]="STEP",s[s.TAN=38]="TAN",s[s.TANH=39]="TANH",s[s.TO_INT=40]="TO_INT"})(pe||(pe={}));const $E="return abs(a);",DE=`
  if (abs(a) > 1.) {
    return uniforms.NAN;
  }
  return acos(a);
`,OE=`
  if (a < 1.) {
    return uniforms.NAN;
  }
  return acosh(a);
`,ME=`
  if (abs(a) > 1.) {
    return uniforms.NAN;
  }
  return asin(a);
`,PE="return asinh(a);",RE=`
  if (isnan(a)) {
    return uniforms.NAN;
  }
  return atan(a);
`,LE=`
  if (abs(a) > 1.) {
    return uniforms.NAN;
  }
  if (a == 1.) {
    return uniforms.INFINITY;
  }
  if (a == -1.) {
    return -uniforms.INFINITY;
  }
  return atanh(a);
`,BE="return ceil(a);",FE="return cos(a);",UE=`
  let e2x = exp(-a);
  return (e2x + 1.0 / e2x) / 2.0;
`,zE="return exp(a) - 1.0;",WE="if (a >= 0.0) { return a; }  return (exp(a) - 1.0);",GE=`
  var resFloat = exp(a) - vec4<f32>(1.0);
  if (a.r >= 0.0) {
    resFloat.r = a.r;
  }
  if (a.g >= 0.0) {
    resFloat.g = a.g;
  }
  if (a.b >= 0.0) {
    resFloat.b = a.b;
  }
  if (a.a >= 0.0) {
    resFloat.a = a.a;
  }
  return resFloat;
`,VE=`
  // Error function is calculated approximately with elementary function.
  // See "Handbook of Mathematical Functions with Formulas,
  // Graphs, and Mathematical Tables", Abramowitz and Stegun.
  let p = ${lv};
  let a1 = ${uv};
  let a2 = ${cv};
  let a3 = ${hv};
  let a4 = ${fv};
  let a5 = ${dv};

  let sign = sign(a);
  let absA = abs(a);
  let t = 1.0 / (1.0 + p * absA);
  return sign * (1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-absA * absA));
`,qE="return exp(a);",HE="return floor(a);",jE="return f32(!isnan(a) && !isinf(a));",KE="return f32(isinf(a));",XE="return f32(isnan(a));",YE="return a;",ZE=`if (a < 0.0) { return uniforms.NAN; }
  return log(a);`,QE=`
  if (isnan(a)) { return a; }
  return log(1.0 + a);
`,JE="return f32(!(a >= 1.0));",eT="return -a;",tT="if (a < 0.0) { return uniforms.alpha * a; } return a;",nT=`
  let aLessThanZero = vec4<f32>(a < vec4<f32>(0.0));
  return (aLessThanZero * (uniforms.alpha * a)) + ((vec4<f32>(1.0) - aLessThanZero) * a);
`,sT="return 1.0 / a;",rT="return select(a, 0.0, a < 0.0);",iT="return clamp(a, 0.0, 6.0);",oT="return clamp(a, vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(6.0, 6.0, 6.0, 6.0));",aT=`
  return select(a, vec4<f32>(0.0), a < vec4<f32>(0.0));
`,lT="return round(a);",uT="return inverseSqrt(a);",cT=`
  if (a >= 0.0) {
    return ${av} * a;
  } else {
    return ${ov} * (exp(a) - 1.0);
  }
`,hT="return 1.0 / (1.0 + exp(-1.0 * a));",fT="return sign(a);",dT="return sin(a);",pT=`
  let e2x = exp(a);
  return (e2x - 1.0 / e2x) / 2.0;
`,mT=`
  let epsilon = 1.1920928955078125e-7;
  let threshold = log(epsilon) + 2.0;

  let too_large = a > -threshold;
  let too_small = a < threshold;
  let exp_a = exp(a);

  if (too_large) {
    return a;
  } else if (too_small) {
    return exp_a;
  } else {
    return log(exp_a + 1.0);
  }
`,gT="return sqrt(a);",yT="return a * a;",bT=`
  if (isnan(a)) {
    return a;
  }

  return select(uniforms.stepAlpha, 1.0, a > 0.0);
`,wT="return tan(a);",xT=`
  let e2x = exp(-2.0 * abs(a));
  return sign(a) * (1.0 - e2x) / (1.0 + e2x);
`,_T="return f32(i32((a)));";function Or(s,e){switch(s){case pe.ABS:return $E;case pe.ACOS:return DE;case pe.ACOSH:return OE;case pe.ASIN:return ME;case pe.ASINH:return PE;case pe.ATAN:return RE;case pe.ATANH:return LE;case pe.COS:return FE;case pe.COSH:return UE;case pe.CEIL:return BE;case pe.ELU:return e?GE:WE;case pe.ERF:return VE;case pe.EXP:return qE;case pe.EXPM1:return zE;case pe.FLOOR:return HE;case pe.IS_FINITE:return jE;case pe.IS_INF:return KE;case pe.IS_NAN:return XE;case pe.LINEAR:return YE;case pe.LOG:return ZE;case pe.LOG1P:return QE;case pe.LOGICAL_NOT:return JE;case pe.NEG:return eT;case pe.LEAKYRELU:return e?nT:tT;case pe.RECIPROCAL:return sT;case pe.RELU:return e?aT:rT;case pe.RELU6:return e?oT:iT;case pe.ROUND:return lT;case pe.RSQRT:return uT;case pe.SELU:return cT;case pe.SIGMOID:return hT;case pe.SIGN:return fT;case pe.SIN:return dT;case pe.SINH:return pT;case pe.SOFTPLUS:return mT;case pe.SQRT:return gT;case pe.SQUARE:return yT;case pe.STEP:return bT;case pe.TAN:return wT;case pe.TANH:return xT;case pe.TO_INT:return _T;default:throw new Error(`BinaryType ${s} is not implemented!`)}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ci(s,e=!1,t=!1,n=3){if(s===null)return"";let r="";if(s==="linear")r=Or(pe.LINEAR);else if(s==="relu")r=Or(pe.RELU,t);else if(s==="elu")r=Or(pe.ELU,t);else if(s==="relu6")r=Or(pe.RELU6,t);else if(s==="prelu")r=NE(Oe.PRELU,t);else if(s==="sigmoid")r=Or(pe.SIGMOID,t);else if(s==="leakyrelu")r=Or(pe.LEAKYRELU,t);else throw new Error(`Activation ${s} has not been implemented for the WebGPU backend.`);const o=xe(t?4:1);let a="";return e?a=`
      fn activation(a : ${o}, coords : vec${n}<i32>) -> ${o} {
        let b = getPreluActivationWeightsByOutputCoords(coords);
        ${r}
      }`:a=`
      fn activation(a : ${o}, coords : vec${n}<i32>) -> ${o} {
        ${r}
      }`,a}function ql(s,e){return`
      ${s?"value = value + getBiasByOutputCoords(coords);":""}
      ${e?"value = activation(value, coords);":""}
      `}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fy(s,e,t=!1,n=!1,r=!1,i=1){R(s&&i===1||!s,()=>`transposeA ${s} is not compatible with component size ${i}`);const o=`
      ${s?"value = getA(batch, col, row);":"value = getA(batch, row, col);"}

    `,a=e?"value = getB(batch, col, row);":"value = getB(batch, row, col);";return`
  fn mm_readA(batch: i32, row: i32, col: i32) -> ${xe(i)} {
    var value = ${xe(i)}(0.0);
    ${t&&r?o:`
    ${s?"if(row < uniforms.dimAOuter && col < uniforms.dimInner)":"if(row < uniforms.aShape[1] && col < uniforms.aShape[2])"}
    {
      ${o}
    }
    `}
    return value;
  }

  fn mm_readB(batch: i32, row: i32, col: i32) -> ${xe(i)} {
    var value = ${xe(i)}(0.0);
    ${a}
    return value;
  }
  `}function ph(s,e,t,n,r=!1,i=!1,o=!1,a=1){return`
  ${fy(t,n,r,i,o,a)}
  fn mm_write(batch: i32, row: i32, col: i32, valueIn: ${xe(a)}) {
    ${r&&i?"":"if (row < uniforms.dimAOuter && col < uniforms.dimBOuter)"}
    {
      var value = valueIn;
      let coords = vec3<i32>(batch, row, col);
      ${ql(s,e)}
      setOutputAtCoords(coords[0], coords[1], coords[2], value);
    }
  }
  `}const vT=(s,e)=>s?`
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          kStart + inputRow,
          globalRowStart + inputCol * ${e});
        `:`
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          globalRow + innerRow,
          kStart + inputCol * ${e});
        `,ST=(s,e,t,n)=>{if(s)return`
      for (var k = 0; k < ${n}; k++) {
        let BCached0 = mm_Bsub[k][tileCol];
        let ACached0 = mm_Asub[k][localRow];
        for (var i = 0; i < ${t}; i++) {
          acc[i] = fma(BCached0, vec4<f32>(ACached0[i]), acc[i]);
        }
      }`;{let r="",i="";for(let o=0;o<e;o++)r+=`let BCached${o} = mm_Bsub[k * ${e} + ${o}][tileCol];`,i+=`acc[i] = fma(BCached${o}, vec4<f32>(ACached[${o}]), acc[i]);`;return`
      for (var k = 0; k < ${n/e}; k++) {
        ${r}
        for (var i = 0; i < ${t}; i++) {
          let ACached = mm_Asub[tileRow + i][k];
          ${i}
        }
      }`}};function mh(s,e,t=!1,n=32,r=!1,i=32,o=!1){const a=e[1]*s[1],l=e[0]*s[0],u=t?a:n,c=t?n:a,h=u/e[0],d=n/e[1],w=s[1],k=s[0];return R((t&&h===4&&s[1]===4||!t&&(h===3||h===4))&&u%e[0]===0&&n%e[1]===0&&s[0]===4,()=>`If transposeA ${t} is true, innerElementSize ${h} and workPerThread[1] ${s[1]} must be 4.
          Otherwise, innerElementSize ${h} must be 3 or 4.
      tileAWidth ${u} must be divisible by workgroupSize[0]${e[0]}. tileInner ${n} must be divisible by workgroupSize[1] ${e[1]}. colPerThread ${s[0]} must be 4.`),`
  var<workgroup> mm_Asub : array<array<vec${h}<f32>, ${u/h}>, ${c}>;
  var<workgroup> mm_Bsub : array<array<vec4<f32>, ${l/s[0]}>, ${n}>;

  ${st()} {
    let localRow = i32(localId.y);
    let tileRow = localRow * ${w};
    let tileCol = i32(localId.x);

    let globalRow = i32(globalId.y) * ${w};
    let globalCol = i32(globalId.x) * ${k};
    let batch = ${r?"0":"i32(globalId.z)"};
    let batchA = ${r||!o?"batch":"batch % uniforms.aShape[0]"};
    let batchB = ${r||!o?"batch":"batch % uniforms.bShape[0]"};
    let globalRowStart = i32(workgroupId.y) * ${a};

    let numTiles = ${r?`${Math.ceil(i/n)}`:`(uniforms.dimInner - 1) / ${n} + 1`};
    var kStart = ${r?`i32(globalId.z) * ${i}`:"0"};

    var acc: array<vec4<f32>, ${w}>;

    // Loop over shared dimension.
    let tileRowB = localRow * ${d};
    for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (var innerRow = 0; innerRow < ${w}; innerRow++) {
            let inputRow = tileRow + innerRow;
            let inputCol = tileCol;
            ${vT(t,h)}
        }

        // Load one tile of B into local memory.
        for (var innerRow = 0; innerRow < ${d}; innerRow++) {
            let inputRow = tileRowB + innerRow;
            let inputCol = tileCol;
            mm_Bsub[inputRow][inputCol] = mm_readB(batchB, kStart + inputRow, globalCol);
        }
        kStart = kStart + ${n};
        workgroupBarrier();

        // Compute acc values for a single thread.
        ${ST(t,h,w,n)}
        workgroupBarrier();
    }

    for (var innerRow = 0; innerRow < ${w}; innerRow++) {
        mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);
    }
  }`}const ud=s=>s?`
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          kStart + inputRow,
          globalRowStart + inputCol);
        `:`
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          globalRowStart + inputRow,
          kStart + inputCol);
        `,kT=s=>s?"let ACached = mm_Asub[k][tileRow + innerRow];":"let ACached = mm_Asub[tileRow + innerRow][k];";function gh(s,e,t=!1,n=32,r=!1,i=32,o=!1,a=!1){const l=s[1]*e[1],u=s[0]*e[0],c=t?l:n,h=t?n:l;R(h%e[1]===0&&c%e[0]===0&&n%e[1]===0,()=>`tileAHight ${h} must be divisible by workgroupSize[1]${e[1]}, tileAWidth ${c} must be divisible by workgroupSize[0]${e[0]}, tileInner ${n} must be divisible by workgroupSize[1]${e[1]}`);const d=h/e[1],w=c/e[0],k=n/e[1],A=s[1],m=s[0],S=o?`
      let localRow = i32(localId.y);
      let localCol = i32(localId.x);
      let globalRowStart = i32(workgroupId.y) * ${l};
      let globalColStart = i32(workgroupId.x) * ${u};

      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (var inputRow = localRow; inputRow < ${h}; inputRow = inputRow + ${e[1]}) {
          for (var inputCol = localCol; inputCol < ${c}; inputCol = inputCol + ${e[0]}) {
            ${ud(t)}
          }
        }
        // Load one tile of B into local memory.
        for (var inputRow = localRow; inputRow < ${n}; inputRow = inputRow + ${e[1]}) {
              for (var inputCol = localCol; inputCol < ${u}; inputCol = inputCol + ${e[0]}) {
            mm_Bsub[inputRow][inputCol] = mm_readB(batchB,
              kStart + inputRow,
              globalColStart + inputCol);
          }
        }
        kStart = kStart + ${n};
        workgroupBarrier();

        // Compute acc values for a single thread.
        var BCached : array<f32, ${m}>;
        for (var k = 0; k < ${n}; k++) {
          for (var inner = 0; inner < ${m}; inner++) {
            BCached[inner] = mm_Bsub[k][localCol + inner * ${e[0]}];
          }
          for (var innerRow = 0; innerRow < ${A}; innerRow++) {
            let ACached = ${t?`mm_Asub[k][localRow + innerRow * ${e[1]}];`:`mm_Asub[localRow + innerRow * ${e[1]}][k];`}
            for (var innerCol = 0; innerCol < ${m}; innerCol++) {
              acc[innerRow][innerCol] =
                  fma(ACached, BCached[innerCol], acc[innerRow][innerCol]);
            }
          }
        }
        workgroupBarrier();
      }
      for (var innerRow = 0; innerRow < ${A}; innerRow++) {
        let gRow = globalRowStart + localRow + innerRow * ${e[1]};
        for (var innerCol = 0; innerCol < ${m}; innerCol++) {
          let gCol = globalColStart + localCol + innerCol * ${e[0]};
          mm_write(batch, gRow, gCol, acc[innerRow][innerCol]);
        }
      }
      `:`
  let tileRow = i32(localId.y) * ${A};
  let tileCol = i32(localId.x) * ${m};

  let globalRow = i32(globalId.y) * ${A};
  let globalCol = i32(globalId.x) * ${m};
  let globalRowStart = i32(workgroupId.y) * ${l};

  let tileRowA = i32(localId.y) * ${d};
  let tileColA = i32(localId.x) * ${w};
  let tileRowB = i32(localId.y) * ${k};
  // Loop over shared dimension.
  for (var t = 0; t < numTiles; t++) {
    // Load one tile of A into local memory.
    for (var innerRow = 0; innerRow < ${d}; innerRow++) {
      for (var innerCol = 0; innerCol < ${w}; innerCol++) {
        let inputRow = tileRowA + innerRow;
        let inputCol = tileColA + innerCol;
        ${ud(t)}
      }
    }

    // Load one tile of B into local memory.
    for (var innerRow = 0; innerRow < ${k}; innerRow++) {
      for (var innerCol = 0; innerCol < ${m}; innerCol++) {
        let inputRow = tileRowB + innerRow;
        let inputCol = tileCol + innerCol;
        mm_Bsub[inputRow][inputCol] = mm_readB(batchB,
          kStart + inputRow,
          globalCol + innerCol);
      }
    }
    kStart = kStart + ${n};
    workgroupBarrier();

    // Compute acc values for a single thread.
    var BCached : array<f32, ${m}>;
    for (var k = 0; k < ${n}; k++) {
      for (var inner = 0; inner < ${m}; inner++) {
        BCached[inner] = mm_Bsub[k][tileCol + inner];
      }

      for (var innerRow = 0; innerRow < ${A}; innerRow++) {
        ${kT(t)}
        for (var innerCol = 0; innerCol < ${m}; innerCol++) {
          acc[innerRow][innerCol] =
              fma(ACached, BCached[innerCol], acc[innerRow][innerCol]);
        }
      }
    }

    workgroupBarrier();
  }

  for (var innerRow = 0; innerRow < ${A}; innerRow++) {
    for (var innerCol = 0; innerCol < ${m}; innerCol++) {
      mm_write(batch, globalRow + innerRow, globalCol + innerCol,
          acc[innerRow][innerCol]);
    }
  }
  `;return`
    var<workgroup> mm_Asub : array<array<f32, ${c}>, ${h}>;
    var<workgroup> mm_Bsub : array<array<f32, ${u}>, ${n}>;

    ${st()} {
      let batch = ${r?"0":"i32(globalId.z)"};
      let batchA = ${r||!a?"batch":"batch % uniforms.aShape[0]"};
      let batchB = ${r||!a?"batch":"batch % uniforms.bShape[0]"};
      let numTiles = ${r?`${Math.ceil(i/n)}`:`(uniforms.dimInner - 1) / ${n} + 1`};
      var kStart = ${r?`i32(globalId.z) * ${i}`:"0"};

      var acc : array<array<f32, ${m}>, ${A}>;

      // Without this initialization strange values show up in acc.
      for (var innerRow = 0; innerRow < ${A}; innerRow++) {
        for (var innerCol = 0; innerCol < ${m}; innerCol++) {
          acc[innerRow][innerCol] = 0.0;
        }
      }
      ${S}
    }
  `}const IT=s=>s?`
      mm_readA(batchA, colA, globalRow),
      mm_readA(batchA, colA + 1, globalRow),
      mm_readA(batchA, colA + 2, globalRow),
      mm_readA(batchA, colA + 3, globalRow)
  `:`
      mm_readA(batchA, globalRow, colA),
      mm_readA(batchA, globalRow, colA + 1),
      mm_readA(batchA, globalRow, colA + 2),
      mm_readA(batchA, globalRow, colA + 3)
  `;function ET(s,e=!1){R(s[1]===1&&s[2]===1,()=>`A linear work group size is required. But got ${s}.`);const t=s[0]*4;return`
    var<workgroup> mm_Asub : array<vec4<f32>, ${s[0]}>;

    ${st()} {
      let tileCol = i32(localId.x);
      let globalCol = i32(globalId.x);
      let globalRow = i32(globalId.y);

      let numTiles = (uniforms.dimInner - 1) / ${t} + 1;
      let batch = i32(globalId.z);
      let batchA = batch % uniforms.aShape[0];
      let batchB = batch % uniforms.bShape[0];
      // Without this initialization strange values show up in acc.
      var acc = 0.0;

      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        let colA = t * ${t} + tileCol * 4;
        mm_Asub[tileCol] = vec4<f32>(${IT(e)});
        workgroupBarrier();

        // Compute acc values for a single thread.
        for (var k = 0; k < ${t/4}; k++) {
          let rowB = t * ${t} + k * 4;
          let BCached = vec4<f32>(mm_readB(batchB, rowB, globalCol),
                              mm_readB(batchB, rowB + 1, globalCol),
                              mm_readB(batchB, rowB + 2, globalCol),
                              mm_readB(batchB, rowB + 3, globalCol));

          let ACached = mm_Asub[k];
          acc = acc + dot(ACached, BCached);
        }

        workgroupBarrier();
      }

      mm_write(batch, globalRow, globalCol, acc);
    }
  `}class TT{constructor(e,t,n=!1,r=!1,i=null,o=null,a=null,l=!1){this.variableNames=["A","B"],this.uniforms="dimAOuter : i32, dimBOuter : i32, dimInner : i32,",this.outputShape=t,this.dispatchLayout={x:[2],y:[1],z:[0]};const u=n?e[1]:e[2];if(this.isVec4=(u%4===0&&!n||t[1]%4===0&&n)&&t[2]%4===0&&!r,this.outputComponent=this.isVec4?4:1,this.isVectorA=t[1]===1&&!n,!this.isVec4&&this.isVectorA)this.elementsPerThread=[1,1,1],this.workgroupSize=[32,1,1];else{const d=Ak(t[1],u,t[2],n);this.workgroupSize=d.workgroupSize,this.elementsPerThread=d.elementsPerThread}this.dispatch=ft(this.dispatchLayout,this.outputShape,this.workgroupSize,this.elementsPerThread);const c=i!=null,h=a!=null;c&&this.variableNames.push("bias"),h&&this.variableNames.push("preluActivationWeights"),this.sequentialAccessByThreads=l,this.transposeA=n,this.transposeB=r,this.addBias=c,this.activation=o,this.hasPreluActivationWeights=h,[this.fitAOuter,this.fitBOuter,this.fitInner]=this.getShapeFit(t[1],t[2],u),this.shaderKey=`matMulPacked_${this.elementsPerThread}_${n}_${r}_${this.activation}_${this.fitAOuter}_${this.fitBOuter}_${this.fitInner}_${this.isVec4}_${this.isVectorA}_${this.sequentialAccessByThreads}`}getShapeFit(e,t,n){const r=this.workgroupSize[1]*this.elementsPerThread[1],i=this.workgroupSize[0]*this.elementsPerThread[0];!this.isVec4&&this.isVectorA?this.tileInner=this.workgroupSize[0]*4:this.tileInner=i;const o=e%r===0,a=t%i===0,l=n%this.tileInner===0;return[o,a,l]}getUserCode(){return`
      ${Ci(this.activation,this.hasPreluActivationWeights,this.isVec4)}
      ${ph(this.addBias,this.activation,!1,this.transposeB,this.fitAOuter,this.fitBOuter,this.fitInner,this.isVec4?4:1)}
      ${this.isVec4?mh(this.elementsPerThread,this.workgroupSize,this.transposeA,this.tileInner,!1,null,!0):this.isVectorA?ET(this.workgroupSize,this.transposeA):gh(this.elementsPerThread,this.workgroupSize,this.transposeA,this.tileInner,!1,null,this.sequentialAccessByThreads,!0)}
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function AT(s,e,t,n,r=!1,i=null,o=!1,a=4,l=4,u=4){const c=$=>{switch($){case 1:return"resData = f32(x[xIndex]);";case 3:return"resData = vec3<f32>(x[xIndex], x[xIndex + 1], x[xIndex + 2]);";case 4:return"resData = vec4<f32>(x[xIndex / 4]);";default:throw new Error(`innerElementSize ${$} is not supported.`)}},h=$=>{switch($){case 1:return"return f32(W[row * uniforms.wShape[3] + col]);";case 4:return"return vec4<f32>(W[(row * uniforms.wShape[3] + col) / 4]);";default:throw new Error(`innerElementSize ${$} is not supported.`)}},d=s?`
      let coord = vec4<i32>(batch, xRow, xCol, xCh);
      `:`
      let coord = vec4<i32>(batch, xCh, xRow, xCol);
      `,w=s?`
      let coords = vec4<i32>(
        batch,
        row / outWidth,
        row % outWidth,
        col);
      `:`
      let coords = vec4<i32>(
        batch,
        row,
        col / outWidth,
        col % outWidth);
      `,k=s?"uniforms.xShape[1]":"uniforms.xShape[2]",A=s?"uniforms.xShape[2]":"uniforms.xShape[3]",m=s?"row":"col",S=s?"col":"row",b=`
      let inChannels = uniforms.wShape[2];
      let outWidth = ${s?"uniforms.outShape[2]":"uniforms.outShape[3]"};
      let outRow = ${m} / outWidth;
      let outCol = ${m} % outWidth;

      let WRow = ${S} / (uniforms.filterDims[1] * inChannels);
      let WCol = ${S} / inChannels % uniforms.filterDims[1];
      let xRow = outRow * uniforms.strides[0] + uniforms.dilations[0] * WRow - uniforms.pads[0];
      let xCol = outCol * uniforms.strides[1] + uniforms.dilations[1] * WCol - uniforms.pads[1];
      let xCh = ${S} % inChannels;
      var resData = ${xe(a)}(0.0);
      // The bounds checking is always needed since we use it to pad zero for
      // the 'same' padding type.
      if (xRow >= 0 && xRow < ${k} && xCol >= 0 && xCol < ${A}) {
        ${d}
        let xIndex = getIndexFromCoords4D(coord, uniforms.xShape);
        ${c(a)}
      }
      return resData;`,f=s?e&&n?`
      ${b}`:`
      if (row < uniforms.dimAOuter && col < uniforms.dimInner) {
        ${b}
      }
      return ${xe(a)}(0.0);`:n&&t?`
      ${b}`:`
      if (row < uniforms.dimInner && col < uniforms.dimBOuter) {
        ${b}
      }
      return ${xe(a)}(0.0);`,v=`${h(l)}`,_=xe(u),E=xe(s?a:l),D=xe(s?l:a);return`
      ${Ci(i,o,u===4,4)}
      fn mm_readA(batch: i32, row : i32, col : i32) -> ${E} {
        ${s?f:v}
      }

      fn mm_readB(batch: i32, row : i32, col : i32) -> ${D} {
        ${s?v:f}
      }

      fn mm_write(batch: i32, row : i32, col : i32, valueIn : ${_}) {
        if (row < uniforms.dimAOuter && col < uniforms.dimBOuter)
        {
        var value = valueIn;
        let outWidth = ${s?"uniforms.outShape[2]":"uniforms.outShape[3]"};
        ${w}
        ${ql(r,i)}
        setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
        }
      }`}class CT{constructor(e,t,n,r,i=!1,o=null,a=!1,l=!1){this.variableNames=["x","W"],this.uniforms="filterDims : vec2<i32>, pads : vec2<i32>, strides : vec2<i32>, dilations : vec2<i32>, dimAOuter : i32, dimBOuter : i32, dimInner : i32,",this.outputShape=e.outShape,this.isChannelsLast=e.dataFormat==="channelsLast",this.isVec4=((e.inChannels%4===0||e.inChannels%3===0)&&this.isChannelsLast||e.outWidth%4===0&&!this.isChannelsLast)&&e.outChannels%4===0,this.dispatchLayout=this.isChannelsLast?{x:[3],y:[1,2],z:[0]}:{x:[2,3],y:[1],z:[0]},this.workgroupSize=Ck(this.dispatchLayout,this.outputShape,this.isVec4),this.elementsPerThread=Nk(this.dispatchLayout,this.outputShape,this.isVec4),this.dispatch=ft(this.dispatchLayout,this.outputShape,this.workgroupSize,this.elementsPerThread),this.isVec4?(this.outputComponent=4,this.isChannelsLast&&e.inChannels%4!==0?(this.innerElementSize=3,this.variableComponents=[1,4]):(this.innerElementSize=4,this.variableComponents=[4,4]),i&&(this.variableNames.push("bias"),this.variableComponents.push(4)),a&&(this.variableNames.push("preluActivationWeights"),this.variableComponents.push(4))):(this.innerElementSize=this.elementsPerThread[0],i&&this.variableNames.push("bias"),a&&this.variableNames.push("preluActivationWeights")),this.sequentialAccessByThreads=l,this.addBias=i,this.activation=o,this.hasPreluActivationWeights=a,this.tileAOuter=this.workgroupSize[1]*this.elementsPerThread[1],this.tileBOuter=this.workgroupSize[0]*this.elementsPerThread[0],this.tileInner=Math.max(this.workgroupSize[0]*this.innerElementSize,this.workgroupSize[1]),this.fitAOuter=t%this.tileAOuter===0,this.fitBOuter=n%this.tileBOuter===0,this.fitInner=r%this.tileInner===0,this.shaderKey=`conv2DMM_${this.elementsPerThread}_${this.activation}}_${this.fitAOuter}_${this.fitBOuter}_${this.fitInner}_${this.isVec4}_${this.innerElementSize}_${this.isChannelsLast}_${this.sequentialAccessByThreads}`}getUserCode(){const e=this.isVec4?mh(this.elementsPerThread,this.workgroupSize,!this.isChannelsLast,this.tileInner):gh(this.elementsPerThread,this.workgroupSize,!this.isChannelsLast,this.tileInner,!1,null,this.sequentialAccessByThreads),t=this.isVec4?[this.innerElementSize,4,4]:[1,1,1];return`
    ${AT(this.isChannelsLast,this.fitAOuter,this.fitBOuter,this.fitInner,this.addBias,this.activation,this.hasPreluActivationWeights,t[0],t[1],t[2])}
    ${e}
  `}}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class NT{constructor(e,t=!1,n=null,r=!1){this.variableNames=["x","W"],this.uniforms="filterDims: vec2<i32>, pads: vec2<i32>, strides: vec2<i32>, dilations: vec2<i32>,",this.workgroupSize=[4,4,8],this.outputShape=e.outShape,this.isChannelsLast=e.dataFormat==="channelsLast",this.dispatchLayout=this.isChannelsLast?{x:[2],y:[1],z:[0,3]}:{x:[3],y:[2],z:[0,1]},this.dispatch=ft(this.dispatchLayout,this.outputShape,this.workgroupSize),this.addBias=t,this.activation=n,this.hasPreluActivationWeights=r,t&&this.variableNames.push("bias"),r&&this.variableNames.push("preluActivationWeights"),this.shaderKey=`conv2dnaive_${this.activation}_${this.isChannelsLast}`}getUserCode(){return`
       ${Ci(this.activation,this.hasPreluActivationWeights,!1,4)}
       fn readInp(batch : i32, row : i32, col : i32, chan : i32) -> f32{
         let coords = vec4<i32>(batch, row, col, chan);
         if (coordsInBounds4D(coords, uniforms.xShape)) {
           return  getX(batch, row, col, chan);
         } else {
          return 0.0;
         }
       }
       fn readFilt(row : i32, col : i32, xChannel : i32, outChannel : i32) -> f32{
         let coords = vec4<i32>(row, col, xChannel, outChannel);
         if(coordsInBounds4D(coords, uniforms.wShape)) {
           return getW(row, col, xChannel, outChannel);
          } else {
            return 0.0;
          }
       }
       fn writeResult(batch : i32, row : i32, col : i32, chan : i32, valueIn : f32) {
         let coords = ${this.isChannelsLast?"vec4<i32>(batch, row, col, chan);":"vec4<i32>(batch, chan, row, col);"}
         if (coordsInBounds4D(coords, uniforms.outShape)) {
           var value = valueIn;
           ${ql(this.addBias,this.activation)}
           setOutputAtCoords(coords.x, coords.y, coords.z, coords.w, value);
         }
       }
       ${st("index")} {
         let coords = getOutputCoords();
         let batch = coords[0];
         let outChannel = ${this.isChannelsLast?"coords[3];":"coords[1];"}
         let outRow = ${this.isChannelsLast?"coords[1];":"coords[2];"}
         let outCol = ${this.isChannelsLast?"coords[2];":"coords[3];"}
         var acc : f32 = 0.0;
         for (var row = 0; row < uniforms.filterDims[0]; row = row + 1) {
           for (var col = 0; col < uniforms.filterDims[1]; col = col + 1) {
             let xRow = outRow * uniforms.strides[0] + uniforms.dilations[0] * row - uniforms.pads[0];
             let xCol = outCol * uniforms.strides[1] + uniforms.dilations[1] * col - uniforms.pads[1];
             for (var xChannel = 0; xChannel < ${this.isChannelsLast?"uniforms.xShape[3];":"uniforms.xShape[1];"} xChannel = xChannel + 1) {
               ${this.isChannelsLast?"let v = readInp(batch, xRow, xCol, xChannel);":"let v = readInp(batch, xChannel, xRow, xCol);"}
               let f = readFilt(row, col, xChannel, outChannel);
               acc = acc + v * f;
             }
           }
         }
         writeResult(batch, outRow, outCol, outChannel, acc);
       }
     `}}/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class $T{constructor(e,t){this.variableNames=["x"],this.uniforms=`pads : vec2<i32>, strides : vec2<i32>, dilations : vec2<i32>, outWidth : i32, itemsPerBlockRow : i32,
       inChannels : i32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=fn(this.outputShape),this.dispatch=ft(this.dispatchLayout,this.outputShape,this.workgroupSize),this.isChannelsLast=t,this.shaderKey=`im2col_${this.isChannelsLast}`}getUserCode(){const e=this.isChannelsLast?1:2,t=this.isChannelsLast?2:3,n=this.isChannelsLast?"coords[1]":"coords[2]",r=this.isChannelsLast?"coords[2]":"coords[1]",i=this.isChannelsLast?"getX(batch, xRow, xCol, ch)":"getX(batch, ch, xRow, xCol)";return`
    ${st("index")} {
      let coords = getCoordsFromIndex(index);
      if(index < uniforms.size) {
        let batch = coords[0];
        let row = ${n};
        let col = ${r};
        let offsetY = (row / uniforms.outWidth) * uniforms.strides[0] - uniforms.pads[0];
        let xRow = offsetY + uniforms.dilations[0] * (col / uniforms.itemsPerBlockRow);
        var value = 0.0;
        if(xRow < uniforms.xShape[${e}] && xRow >= 0) {
          let offsetX = (row % uniforms.outWidth) * uniforms.strides[1] -
              uniforms.pads[1];
          let xCol = offsetX + uniforms.dilations[1] * ((col %
              uniforms.itemsPerBlockRow) / uniforms.inChannels);
          let ch = col % uniforms.inChannels;
          if(xCol < uniforms.xShape[${t}] && xCol >= 0) {
            value = ${i};
          }
        }
        setOutputAtIndex(index, value);
      }
    }
   `}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function DT(s){return`
    var<workgroup> sumValues : array<f32, ${s}>;
    ${st()} {
      let coords = getOutputCoords();
      let batch = coords[0];
      let batchA = batch % uniforms.aShape[0];
      let batchB = batch % uniforms.bShape[0];
      let row = coords[1];
      let col = coords[2];
      var sum = 0.0;
      let Length = uniforms.dimInner;
      for (var k = i32(localId.x); k < Length; k = k + ${s}) {
        let dataA = mm_readA(batchA, row, k);
        let dataB = mm_readB(batchB, k, col);
        sum = sum + dataA * dataB;
      }
      sumValues[localId.x] = sum;
      workgroupBarrier();

      for(var currentSize = ${s/2}u; currentSize > 1u;
          currentSize = currentSize / 2u) {
        if (localId.x < currentSize)
        {
          sumValues[localId.x] = sumValues[localId.x] + sumValues[localId.x + currentSize];
        }
        workgroupBarrier();
      }

      if (localId.x == 0u) {
        sum = sumValues[0] + sumValues[1];
        mm_write(batch, row, col, sum);
      }
    }
  `}class OT{constructor(e,t=!1,n=!1,r=null,i=null,o=null){this.variableNames=["A","B"],this.uniforms="dimAOuter : i32, dimBOuter : i32, dimInner : i32,",this.workgroupSize=[256,1,1],this.outputShape=e,this.dispatchLayout={x:[],y:[1,2],z:[0]},this.dispatch=ft(this.dispatchLayout,this.outputShape,this.workgroupSize);const a=r!=null,l=o!=null;a&&this.variableNames.push("bias"),l&&this.variableNames.push("preluActivationWeights"),this.transposeA=t,this.transposeB=n,this.addBias=a,this.activation=i,this.hasPreluActivationWeights=l,this.shaderKey=`matMulReduce_${this.activation}_${t}_${n}`}getUserCode(){return`
      ${Ci(this.activation,this.hasPreluActivationWeights)}
      ${ph(this.addBias,this.activation,this.transposeA,this.transposeB)}
      ${DT(this.workgroupSize[0])}
    `}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function MT(s){const e=s[1],t=s[0],n=e>t?e:t;return`
  var<workgroup> mm_Asub : array<array<f32, ${n}>, ${e}>;
  var<workgroup> mm_Bsub : array<array<f32, ${t}>, ${n}>;

  // If the output size is small for matrix multiplication, avoid to use vec4
  // and handle some elements per thread to optimally utilize the ALU.
  // Read data from global memory to registers firstly, then store them into
  // shared memory, so it is instruction-Level parallelism for arithmetic
  // operations and others handle IO operations between barrier api, makes ALU
  // and load/store units work simultaneously, could improves the performance.
  ${st()} {
    let tileRow = i32(localId.y);
    let tileCol = i32(localId.x);
    let globalRow = i32(globalId.y);
    let globalCol = i32(globalId.x);
    let batch = i32(globalId.z);
    let batchA = batch % uniforms.aShape[0];
    let batchB = batch % uniforms.bShape[0];

    // uniforms.dimInner should be greater than 0.
    let numTiles = (uniforms.dimInner - 1) / ${n} + 1;
    var acc = 0.0;

    var globalColA = tileCol;
    var globalRowB = 0;
    var regA = mm_readA(batchA, globalRow, globalColA);
    var regB0 = mm_readB(batchB, globalRowB + 2 * tileRow, globalCol);
    var regB1 = mm_readB(batchB, globalRowB + 2 * tileRow + 1, globalCol);
    globalColA = globalColA + ${n};
    globalRowB = globalRowB + ${n};

    for (var t = 0; t < numTiles; t = t + 1) {
      mm_Asub[tileRow][tileCol] = regA;
      mm_Bsub[2 * tileRow][tileCol] = regB0;
      mm_Bsub[2 * tileRow + 1][tileCol] = regB1;

      workgroupBarrier();

      regA = mm_readA(batchA, globalRow, globalColA);
      regB0 = mm_readB(batchB, globalRowB + 2 * tileRow, globalCol);
      regB1 = mm_readB(batchB, globalRowB + 2 * tileRow + 1, globalCol);
      globalColA = globalColA + ${n};
      globalRowB = globalRowB + ${n};

      for (var k = 0; k < ${n}; k = k + 1) {
        acc = acc + mm_Asub[tileRow][k] * mm_Bsub[k][tileCol];
      }
      workgroupBarrier();
    }

    mm_write(batch, globalRow, globalCol, acc);
  }
  `}class PT{constructor(e,t,n,r=!1,i=!1,o=null,a=null,l=null){this.variableNames=["A","B"],this.uniforms="dimAOuter : i32, dimBOuter : i32, dimInner : i32,",this.workgroupSize=[16,8,1],this.outputShape=n,this.dispatchLayout={x:[2],y:[1],z:[0]},this.dispatch=[Math.ceil(n[2]/this.workgroupSize[0]),Math.ceil(n[1]/this.workgroupSize[1]),n[0]];const u=o!=null;u&&this.variableNames.push("bias");const c=l!=null;c&&this.variableNames.push("preluActivationWeights"),this.transposeA=r,this.transposeB=i,this.addBias=u,this.activation=a,this.hasPreluActivationWeights=c,this.shaderKey=`matMulSmallOutputSize_${this.activation}_${r}_${i}`}getUserCode(){return`
      ${Ci(this.activation,this.hasPreluActivationWeights)}
      ${ph(this.addBias,this.activation,this.transposeA,this.transposeB)}
      ${MT(this.workgroupSize)}
    `}}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class RT{constructor(e,t,n=!1,r=!1){this.variableNames=["A","B"],this.uniforms="dimAOuter : i32, dimBOuter : i32, dimInner : i32,",this.workgroupSize=[8,8,1],this.atomic=!0,this.splitedDimInner=128,R(e[0]===1,()=>"MatMulSplitKProgram only supports batch = 1."),this.outputShape=e,this.dispatchLayout={x:[2],y:[1],z:[0,3]};const i=(n&&this.outputShape[1]%4===0||!n&&t%4===0)&&this.outputShape[2]%4===0;this.elementsPerThread=[4,4,this.splitedDimInner],this.outputComponent=i?4:1,i||(this.outputShape[1]<16&&(this.elementsPerThread[1]=1),this.outputShape[2]<16&&(this.elementsPerThread[0]=1)),this.dispatch=ft(this.dispatchLayout,[this.outputShape[0],this.outputShape[1],this.outputShape[2],t],this.workgroupSize,this.elementsPerThread),this.transposeA=n,this.transposeB=r,this.shaderKey=`matMulSplitK_${n}_${r}_${this.elementsPerThread}_${this.outputComponent}`}getUserCode(){const e=this.outputComponent;return`
      ${fy(!1,this.transposeB,!1,!1,!1,e)}
      fn mm_write(batch: i32, row : i32, col : i32, value : ${xe(e)}) {
        if (row < uniforms.dimAOuter && col < uniforms.dimBOuter) {
          let coords = vec3<i32>(batch, row, col);
          let flatIndex = getOutputIndexFromCoords(coords);
          // The problem is that we should initialize output to zero before using.
          // Otherwise, the original value will be added to the result.
          for (var i = 0; i < ${e}; i = i + 1) {
            ${mk("&result[flatIndex + i]",`${e>1?"value[i]":"value"}`)}
          }
        }
      }
      ${e===4?mh(this.elementsPerThread,this.workgroupSize,this.transposeA,32,!0,this.splitedDimInner):gh(this.elementsPerThread,this.workgroupSize,this.transposeA,32,!0,this.splitedDimInner)}
    `}}class LT{constructor(e,t=null,n=null,r=null){this.uniforms="",this.variableNames=["x"],this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=fn(this.outputShape),this.dispatch=ft(this.dispatchLayout,this.outputShape,this.workgroupSize),this.addBias=t!=null,this.hasPreluActivationWeights=r!=null,this.activation=n,this.addBias&&this.variableNames.push("bias"),this.hasPreluActivationWeights&&this.variableNames.push("preluActivationWeights"),this.shaderKey=`biasActivation_${n}`}getUserCode(){return`
    ${Ci(this.activation,this.hasPreluActivationWeights)}
    ${st("index")} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        var value = getXByOutputIndex(index);
        ${ql(this.addBias,this.activation)}
        setOutputAtIndex(index, value);
      }
    }
    `}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ve(s){const{inputs:e,attrs:t}=s,{x:n}=e,{shape:r}=t,i=me(n.shape),o=m0(r,i),a=me(o);return R(i===a,()=>`The new shape (${o}) has ${a} elements and the old shape (${n.shape}) has ${i} elements. The new shape and old shape must have the same number of elements.`),s.backend.incRef(n.dataId),{dataId:n.dataId,shape:o,dtype:n.dtype}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dy({a:s,b:e,transposeA:t,transposeB:n,backend:r,bias:i=null,preluActivationWeights:o=null,leakyreluAlpha:a=0,activation:l=null}){const u=s.shape.length,c=e.shape.length,h=t?s.shape[u-2]:s.shape[u-1],d=e.shape[c-2],w=t?s.shape[u-1]:s.shape[u-2],k=e.shape[c-1],A=s.shape.slice(0,-2),m=e.shape.slice(0,-2),S=me(A),b=me(m),v=Ft(s.shape.slice(0,-2),e.shape.slice(0,-2)).concat([w,k]);R(h===d,()=>`Error in matMul: inner shapes (${h}) and (${d}) of Tensors with shapes ${s.shape} and ${e.shape} and transposeA=${t} and transposeB=${n} must match.`);const _=t?[S,h,w]:[S,w,h],E=[b,d,k],D=Ve({inputs:{x:s},backend:r,attrs:{shape:_}}),M=Ve({inputs:{x:e},backend:r,attrs:{shape:E}}),$=[D,M],C=Math.max(S,b),g=[D,M],p=[{type:"int32",data:[w]},{type:"int32",data:[k]},{type:"int32",data:[h]}];let y,x;const I=[C,w,k];let N=ge().get("WEBGPU_MATMUL_PROGRAM_TYPE");if(N<0){const W=ge().getNumber("WEBGPU_THRESHOLD_TO_INCREASE_WORKGROUPS_FOR_MATMUL"),X=W>0?W:r.thresholdToIncreaseWorkgroups,V=C*Math.ceil(w/32)*Math.ceil(k/32);V<=X||w<=8&&V<=X*2?C*w*k<=128?N=Vn.MatMulReduceProgram:C===1&&d>=2e3?N=Vn.MatMulSplitKProgram:N=Vn.MatMulSmallOutputSizeProgram:N=Vn.MatMulPackedProgram}switch(N){case Vn.MatMulReduceProgram:y=new OT(I,t,n,i,l,o);break;case Vn.MatMulSplitKProgram:{if(x=uy({backend:r,attrs:{shape:I,value:0,dtype:s.dtype}}),y=new RT(I,d,t,n),i||l){x=r.runWebGPUProgram(y,g,s.dtype,p,x);const X=new LT(x.shape,i,l,o);let V=null;const Z=[x];i&&Z.push(i),o&&Z.push(o),l==="leakyrelu"&&(V=[{type:"float32",data:[a]}],X.uniforms+=" alpha : f32,");const te=r.runWebGPUProgram(X,Z,x.dtype,V);$.push(x);const oe=Ve({inputs:{x:te},backend:r,attrs:{shape:v}});$.push(te);for(const ce of $)r.disposeData(ce.dataId);return oe}break}case Vn.MatMulSmallOutputSizeProgram:y=new PT(_,E,I,t,n,i,l,o);break;case Vn.MatMulPackedProgram:const W=r.adapterInfo.isIntel();y=new TT(_,I,t,n,i,l,o,W);break;default:throw new Error(`Unsupported MatMulProgramType ${N}.`)}i&&g.push(i),o&&g.push(o),l==="leakyrelu"&&(p.push({type:"float32",data:[a]}),y.uniforms+=" alpha : f32,"),x=r.runWebGPUProgram(y,g,s.dtype,p,x);const L=Ve({inputs:{x},backend:r,attrs:{shape:v}});$.push(x);for(const W of $)r.disposeData(W.dataId);return L}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ml(s,e){const t=s.length;return t>=3?e?[...s.slice(0,-3),s[t-3]*s[t-2],s[t-1]]:[...s.slice(0,-3),s[t-3],s[t-2]*s[t-1]]:!e&&t===1&&s[0]>1?[s[0],1]:null}function BT({x:s,filter:e,convInfo:t,backend:n,bias:r=null,preluActivationWeights:i=null,leakyreluAlpha:o=0,activation:a=null}){const l=t.dataFormat==="channelsLast",u=!l,c=!1,h=l&&t.filterHeight===t.inHeight&&t.filterWidth===t.inWidth&&t.padInfo.type==="VALID",d=[];let w,k;if(h){const S=t.inHeight*t.inWidth*t.inChannels;w=Ve({inputs:{x:s},backend:n,attrs:{shape:[1,t.batchSize,S]}}),k=Ve({inputs:{x:e},backend:n,attrs:{shape:[1,S,t.outChannels]}})}else w=Ve({inputs:{x:s},backend:n,attrs:{shape:l?[t.batchSize,t.inHeight*t.inWidth,t.inChannels]:[t.batchSize,t.inChannels,t.inHeight*t.inWidth]}}),k=Ve({inputs:{x:e},backend:n,attrs:{shape:[1,t.inChannels,t.outChannels]}});if(d.push(w),d.push(k),i!=null){const S=ml(i.shape,l);S!=null&&(i=Ve({inputs:{x:i},backend:n,attrs:{shape:S}}),d.push(i))}if(r!=null){const S=ml(r.shape,l);S!=null&&(r=Ve({inputs:{x:r},backend:n,attrs:{shape:S}}),d.push(r))}const A=dy({a:l?w:k,b:l?k:w,transposeA:u,transposeB:c,backend:n,bias:r,activation:a,preluActivationWeights:i,leakyreluAlpha:o}),m=Ve({inputs:{x:A},backend:n,attrs:{shape:t.outShape}});d.push(A);for(const S of d)n.disposeData(S.dataId);return m}function FT({x:s,filter:e,convInfo:t,backend:n,bias:r=null,preluActivationWeights:i=null,leakyreluAlpha:o=0,activation:a=null}){const{filterWidth:l,filterHeight:u,inChannels:c,strideWidth:h,strideHeight:d,padInfo:w,outWidth:k,outHeight:A,dilationWidth:m,dilationHeight:S,dataFormat:b}=t,f=b==="channelsLast",v=l*u*c,_=A*k,E=f?[t.batchSize,_,v]:[t.batchSize,v,_],D=new $T(E,f),M=[{type:"int32",data:[w.top,w.left]},{type:"int32",data:[d,h]},{type:"int32",data:[S,m]},{type:"int32",data:[k]},{type:"int32",data:[c*l]},{type:"int32",data:[c]}],$=n.runWebGPUProgram(D,[s],s.dtype,M),C=[];C.push($);const g=Ve({inputs:{x:e},backend:n,attrs:{shape:[1,v,-1]}});if(C.push(g),i!=null){const N=ml(i.shape,f);N!=null&&(i=Ve({inputs:{x:i},backend:n,attrs:{shape:N}}),C.push(i))}if(r!=null){const N=ml(r.shape,f);N!=null&&(r=Ve({inputs:{x:r},backend:n,attrs:{shape:N}}),C.push(r))}const x=dy({a:f?$:g,b:f?g:$,transposeA:!f,transposeB:!1,backend:n,bias:r,activation:a,preluActivationWeights:i,leakyreluAlpha:o}),I=Ve({inputs:{x},backend:n,attrs:{shape:t.outShape}});C.push(x);for(const N of C)n.disposeData(N.dataId);return I}function UT({x:s,filter:e,convInfo:t,backend:n,bias:r=null,preluActivationWeights:i=null,leakyreluAlpha:o=0,activation:a=null}){const l=r!=null,u=i!=null,c=t.dataFormat==="channelsLast",h=c&&t.filterHeight===t.inHeight&&t.filterWidth===t.inWidth&&t.padInfo.type==="VALID",d=ge().getBool("WEBGPU_USE_NAIVE_CONV2D_DEBUG");if(!d&&(h||t.filterHeight===1&&t.filterWidth===1&&t.dilationHeight===1&&t.dilationWidth===1&&t.strideHeight===1&&t.strideWidth===1&&(t.padInfo.type==="SAME"||t.padInfo.type==="VALID")))return BT({x:s,filter:e,convInfo:t,backend:n,bias:r,activation:a,preluActivationWeights:i,leakyreluAlpha:o});const w=ge().getNumber("WEBGPU_THRESHOLD_TO_INCREASE_WORKGROUPS_FOR_MATMUL"),k=w>-1?w:n.thresholdToIncreaseWorkgroups,A=t.batchSize*Math.ceil(t.outHeight*t.outWidth/32)*Math.ceil(t.outChannels/32);if(ge().getBool("WEBGPU_CONV_SEPARATE_IM2COL_SHADER")||A<=k)return FT({x:s,filter:e,convInfo:t,backend:n,bias:r,preluActivationWeights:i,leakyreluAlpha:o,activation:a});let m;const S=[t.padInfo.top,t.padInfo.left],b=[{type:"int32",data:[t.filterHeight,t.filterWidth]},{type:"int32",data:[...S]},{type:"int32",data:[t.strideHeight,t.strideWidth]},{type:"int32",data:[t.dilationHeight,t.dilationWidth]}];if(d)m=new NT(t,l,a,u);else{const E=c?t.outHeight*t.outWidth:t.outChannels,D=c?t.outChannels:t.outHeight*t.outWidth,M=t.filterHeight*t.filterWidth*t.inChannels;b.push({type:"int32",data:[E]},{type:"int32",data:[D]},{type:"int32",data:[M]});const $=n.adapterInfo.isIntel();m=new CT(t,E,D,M,l,a,u,$)}const f=[],v=[s,e];l&&(!c&&r.shape.length===1&&(r=Ve({inputs:{x:r},backend:n,attrs:{shape:[r.shape[0],1,1]}}),f.push(r)),v.push(r)),u&&(!c&&i.shape.length===1&&(i=Ve({inputs:{x:i},backend:n,attrs:{shape:[i.shape[0],1,1]}}),f.push(i)),v.push(i)),a==="leakyrelu"&&(b.push({type:"float32",data:[o]}),m.uniforms+=" alpha : f32,");const _=n.runWebGPUProgram(m,v,s.dtype,b);for(const E of f)n.disposeData(E.dataId);return _}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zT(s){const{inputs:e,backend:t,attrs:n}=s,{x:r,filter:i,bias:o,preluActivationWeights:a}=e,{strides:l,pad:u,dataFormat:c,dilations:h,dimRoundingMode:d,activation:w,leakyreluAlpha:k}=n,A=xw(c),m=Sc(r.shape,i.shape,l,h,u,d,!1,A);return UT({x:r,filter:i,convInfo:m,backend:t,bias:o,preluActivationWeights:a,leakyreluAlpha:k,activation:w})}const WT={kernelName:Au,backendName:"webgpu",kernelFunc:zT};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class GT{constructor(e){this.variableNames=["x"],this.uniforms="strides : vec2<i32>,",this.workgroupSize=[256,1,1],this.size=!0,this.outputShape=e.outShape,this.dispatchLayout=fn(this.outputShape),this.dispatch=ft(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey="poolWithFilterSizeEqualsOne"}getUserCode(){return`
      ${st("index")} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let batch = coords[0];
          let d = coords[3];

          let xRCCorner = coords.yz * uniforms.strides;
          let xRCorner = xRCCorner.x;
          let xCCorner = xRCCorner.y;

          let value = getX(batch, xRCorner, xCCorner, d);
          setOutputAtIndex(index, value);
        }
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class VT{constructor(e,t,n=!1,r=!1,i=!1){if(this.variableNames=["x"],this.uniforms="strides : vec2<i32>, pads : vec2<i32>, dilations : vec2<i32>, convDims : vec2<i32>, filterDims : vec2<i32>,",this.workgroupSize=[128,1,1],this.size=!0,t==="avg"&&n)throw new Error("Cannot compute positions for average pool.");this.outputShape=e.outShape,this.dispatchLayout=fn(this.outputShape),this.dispatch=ft(this.dispatchLayout,this.outputShape,this.workgroupSize),this.poolType=t,this.computePositions=n,this.flattenPositions=r,this.includeBatchIndex=i,this.shaderKey=`pool2D_${t}_${n}_${r}_${i}`}getUserCode(){let e;this.poolType==="avg"?e="resultValue = resultValue + value; count = count + 1.0;":this.computePositions?e=`let currMaxValue = mix(value, maxValue, maxValueFound);
      if (value >= currMaxValue) {
        maxValue = value;
        maxValueFound = 1.0;
        maxPosition = ${this.flattenPositions?this.includeBatchIndex?"((batch * uniforms.xShape[1] + xR) * uniforms.xShape[2] + xC) * uniforms.xShape[3] + d":"(xR * uniforms.xShape[2] + xC) * uniforms.xShape[3] + d":"wR * uniforms.filterDims.y + wC"};
      }`:e="resultValue = max(value, resultValue);";let t="resultValue";return this.poolType==="avg"&&(t="resultValue / max(count, 1.0)"),`
      ${st("index")} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
          let batch = coords[0];
          let d = coords[3];
          let xRCCorner = vec2<i32>(coords.yz) * uniforms.strides - uniforms.pads;
          let xRCorner = xRCCorner.x;
          let xCCorner = xRCCorner.y;

          ${this.computePositions?`var maxValue = 0.0;
            var maxValueFound = 0.0;
            var maxPosition = 0;`:`var resultValue = ${this.poolType==="avg"?"0.0":"-1.0 / pow(10.0, -20.0)"};`}

          var count = 0.0;
          for (var wR = 0; wR < uniforms.filterDims.x; wR = wR + uniforms.dilations.x) {
            let xR = xRCorner + wR;

            if (xR < 0 || xR >= uniforms.convDims.x) {
              continue;
            }

            for (var wC = 0; wC < uniforms.filterDims.y; wC = wC + uniforms.dilations.y) {
              let xC = xCCorner + wC;
              if (xC < 0 || xC >= uniforms.convDims.y) {
                continue;
              }

              let value = getX(batch, xR, xC, d);
              ${e}
            }
          }

          ${this.computePositions?"setOutputAtIndexI32(index, maxPosition);":`setOutputAtIndex(index, ${t});`}
        }
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class qT{constructor(e,t){this.variableNames=["A"],this.workgroupSize=[16,16,1];const n=new Array(e.length);for(let r=0;r<n.length;r++)n[r]=e[t[r]];this.outputShape=n,this.dispatchLayout={x:[0],y:[1]},this.dispatch=ft(this.dispatchLayout,this.outputShape,this.workgroupSize,[1,1,1]),this.shaderKey="transposeShared"}getUserCode(){R(this.workgroupSize[0]===this.workgroupSize[1],()=>`Must be a square tile, current tile shape is ${this.workgroupSize[0]} x ${this.workgroupSize[1]}`);const e=this.workgroupSize[0];return`
      var<workgroup> tile : array<array<f32, ${this.workgroupSize[0]+1}>, ${this.workgroupSize[0]}>;
      ${st()} {
        var x = i32(workgroupId.x) * ${e} + i32(localId.x);
        var y = i32(workgroupId.y) * ${e} + i32(localId.y);
        let width = uniforms.outShape[0];
        let height = uniforms.outShape[1];
        if (x < width && y < height) {
          tile[localId.y][localId.x] = f32(A[y * width + x]);
        }
        workgroupBarrier();

        x = i32(workgroupId.y) * ${e} + i32(localId.x);
        y = i32(workgroupId.x) * ${e} + i32(localId.y);
        if (x < height && y < width) {
          setOutputAtIndex((y * height + x), tile[localId.x]
            [localId.y]);
        }
      }
    `}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class HT{constructor(e,t){this.variableNames=["A"],this.workPerThread=1,this.workgroupSize=[64,1,1],this.size=!0;const n=new Array(e.length);for(let r=0;r<n.length;r++)n[r]=e[t[r]];this.outputShape=n,this.dispatchLayout=fn(this.outputShape),this.dispatch=ft(this.dispatchLayout,this.outputShape,this.workgroupSize,[this.workPerThread,1,1]),this.newDim=t,this.shaderKey=`transpose_${t}`}getUserCode(){const e=kt(this.outputShape.length),t=jT(this.newDim);return`
      ${st("index")} {
        for(var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let flatIndex = index * ${this.workPerThread} + i;
          if(flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndex);
            setOutputAtIndex(flatIndex, A[getIndexFromCoords${this.outputShape.length}D(
              ${e}(${t}), uniforms.aShape)]);
          }
        }
      }
    `}}function jT(s){const e=s.length;if(e>6)throw Error(`Transpose for rank ${e} is not yet supported`);const t=new Array(e);for(let n=0;n<s.length;n++)t[s[n]]=`coords.${wr(n)}`;return t.join()}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function KT(s){const{inputs:e,backend:t,attrs:n}=s,{x:r}=e,{perm:i}=n,o=t,a=r.shape.length,l=new Array(a);for(let c=0;c<l.length;c++)l[c]=r.shape[i[c]];if(t.shouldExecuteOnCPU([r])){const h=o.tensorMap.get(r.dataId).values,d=YI(h,r.shape,r.dtype,i,l);return t.makeTensorInfo(l,r.dtype,d)}if(r.shape.length===2&&cn(i,[1,0])){const c=new qT(r.shape,i);return o.runWebGPUProgram(c,[r],r.dtype)}const u=new HT(r.shape,i);return o.runWebGPUProgram(u,[r],r.dtype)}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class XT{constructor(e,t,n){this.variableNames=["x"],this.uniforms="reduceSize : i32,",this.size=!0,this.inputShape=[e.batchSize,e.inSize];const[r]=Tc(this.inputShape,[1]);this.outputShape=r.length===0?[1]:r,e.inSize>=32768&&n>=512?this.workgroupSize=[512,1,1]:e.inSize>=4096?this.workgroupSize=[256,1,1]:this.workgroupSize=[64,1,1],this.dispatchLayout=fn(this.outputShape),this.dispatch=ft(this.dispatchLayout,this.outputShape,[1,1,1]),this.reduceType=t,this.shaderKey=`reduce_${t}`}getUserCode(){let e="",t="0.0";const n=this.workgroupSize[0];this.reduceType==="min"||this.reduceType==="max"?(e=`
         if (isnan(candidate)) {
          bestValue = uniforms.NAN;
         } else if (!isnan(bestValue) && candidate ${this.reduceType==="min"?"<":">"} bestValue)
           {  bestValue = candidate; }`,t="f32(x[offset])"):this.reduceType==="sum"||this.reduceType==="mean"?e=" bestValue = bestValue + candidate; ":this.reduceType==="prod"?(e=" bestValue = bestValue * candidate; ",t="1.0"):this.reduceType==="all"?(e=" bestValue = f32(bestValue >= 1.0 && candidate >= 1.0); ",t="1.0"):this.reduceType==="any"&&(e=" bestValue = f32(bestValue >= 1.0 || candidate >= 1.0); ",t="0.0");const r=this.reduceType==="mean"?"setOutputAtIndex(outputIndex, bestValue / f32(uniforms.reduceSize));":"setOutputAtIndex(outputIndex, bestValue);";return`
       fn DIV_CEIL(a : u32, b : u32) -> u32 {
        return ((a - 1u) / b + 1u);
       }

       ${`
         var<workgroup> xBestValues : array<f32, ${n}>;
       `}
       fn getOffset(outputIndex : i32) -> i32 {
         let outputCoords = getCoordsFromIndex(outputIndex);
         let offset = ${this.outputShape.length===1?"outputCoords":"outputCoords[0]"} * uniforms.reduceSize;
          return offset;
       }
       ${st("index")} {
         let outputIndex = index / ${n};
         let offset = getOffset(outputIndex);
         var bestValue = ${t};
         let Length = uniforms.reduceSize;
         let WorkPerThread = DIV_CEIL(u32(Length), ${n}u);
         for (var k = i32(localId.x); k < Length && outputIndex < uniforms.size;
             k = k + ${n}) {
           let candidate = f32(x[offset + k]);
           ${e}
         }
         xBestValues[localId.x] = bestValue;
         workgroupBarrier();

         var reduceSize = min(u32(Length), ${n}u);
         for (var currentSize = reduceSize / 2u; reduceSize > 1u;
             currentSize = reduceSize / 2u) {
           let interval = DIV_CEIL(reduceSize, 2u);
           if (localId.x < currentSize) {
            let candidate = xBestValues[localId.x + interval];
            ${e}
            xBestValues[localId.x] = bestValue;
           }
           reduceSize = interval;
           workgroupBarrier();
         }

         if (localId.x == 0u && outputIndex < uniforms.size) {
          ${r}
        }
       }
     `}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const YT={mean:"float32",all:"bool",any:"bool"};function ZT(s,e,t,n,r){const i=s.shape.length,o=[],a=Co(e,s.shape);let l=a;const u=sx(l,i);let c=s;u!=null&&(c=KT({inputs:{x:s},attrs:{perm:u},backend:r}),l=rx(l.length,i),o.push(c)),nx(n,l,i);const[h,d]=Tc(c.shape,l);let w=h;t&&(w=Dp(h,a));let k;if(r.shouldExecuteOnCPU([c])){const A=r.tensorMap.get(c.dataId).values;switch(n){case"max":const m=jI(A,me(d),w,s.dtype);k=r.makeTensorInfo(w,s.dtype,m);break;case"prod":const{outVals:S,outShape:b,outDtype:f}=KI(c.shape,c.dtype,A,l);k=r.makeTensorInfo(b,f,S);break;default:throw new Error(`${n} CPU implementation is not yet supported.`)}}else{const A=me(d),S=me(c.shape)/A,b={windowSize:A,inSize:A,batchSize:S,outSize:1},f=YT[n]||w1(s.dtype),v=[{type:"int32",data:[A]}],_=new XT(b,n,r.device.limits.maxComputeWorkgroupSizeX),E=r.runWebGPUProgram(_,[c],f,v);o.push(E),k=Ve({inputs:{x:E},attrs:{shape:w},backend:r})}return o.forEach(A=>r.disposeData(A.dataId)),k}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function QT(s){const{inputs:e,backend:t,attrs:n}=s,{x:r}=e,{reductionIndices:i,keepDims:o}=n;return ZT(r,i,o,"max",t)}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function JT(s,e,t,n){if(e.filterWidth===1&&e.filterHeight===1&&cn(e.inShape,e.outShape))return Ls({inputs:{x:s},backend:n});if(e.filterWidth===e.inWidth&&e.filterHeight===e.inHeight&&e.batchSize===1&&e.padInfo.type==="VALID"){const o=s.shape.length,a=Ve({inputs:{x:s},backend:n,attrs:{shape:[s.shape[o-3]*s.shape[o-2],s.shape[o-1]]}});let l;R(t==="max",()=>`Invalid pool type ${t}`),l=QT({inputs:{x:a},backend:n,attrs:{reductionIndices:0,keepDims:!1}});const u=Ve({inputs:{x:l},backend:n,attrs:{shape:e.outShape}});return n.disposeData(a.dataId),n.disposeData(l.dataId),u}let r;const i=[{type:"int32",data:[e.strideHeight,e.strideWidth]}];return e.filterHeight===1&&e.filterWidth===1?r=new GT(e):(R(t==="max",()=>`Invalid pool type ${t}`),r=new VT(e,"max"),i.push({type:"int32",data:[e.padInfo.top,e.padInfo.left]},{type:"int32",data:[e.dilationHeight,e.dilationWidth]},{type:"int32",data:[e.inHeight,e.inWidth]},{type:"int32",data:[e.effectiveFilterHeight,e.effectiveFilterWidth]})),n.runWebGPUProgram(r,[s],s.dtype,i)}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function eA(s){const{inputs:e,backend:t,attrs:n}=s,{x:r}=e,{filterSize:i,strides:o,pad:a,dimRoundingMode:l}=n,c=gw(r.shape,i,o,1,a,l);return JT(r,c,"max",t)}const tA={kernelName:np,backendName:"webgpu",kernelFunc:eA};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class nA{constructor(e,t,n,r){this.variableNames=["x"],this.uniforms="adjustHeightWidth : vec2<f32>, roundBase : f32,",this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=[e[0],t,n,e[3]],this.dispatchLayout=fn(this.outputShape),this.dispatch=ft(this.dispatchLayout,this.outputShape,this.workgroupSize),this.halfPixelCenters=r,this.shaderKey=`resizeNearest_${r}`}getUserCode(){let e;return this.halfPixelCenters?e="max((vec2<f32>(rc) + vec2<f32>(0.5)) * effectiveInputOverOutputRatioRC, vec2<f32>(0.0))":e="vec2<f32>(rc) * effectiveInputOverOutputRatioRC",`
      ${st("index")} {
        if (index < uniforms.size) {
          let coords = getCoordsFromIndex(index);
          let b = coords[0];
          let d = coords[3];
          let rc = coords.yz;

          let effectiveInSize = vec2<f32>(
            f32(uniforms.xShape.y) - uniforms.adjustHeightWidth[0],
            f32(uniforms.xShape.z) - uniforms.adjustHeightWidth[1]);

          let effectiveOutSize = vec2<f32>(
            f32(uniforms.outShape.y) - uniforms.adjustHeightWidth[0],
            f32(uniforms.outShape.z) - uniforms.adjustHeightWidth[1]);

          let effectiveInputOverOutputRatioRC =
              effectiveInSize / effectiveOutSize;

          // Fractional source index
          let sourceFracIndexRC = ${e};

          // Compute the coordinators of nearest neighbor point.
          let inputShapeRC = vec2<f32>(f32(uniforms.xShape.y), f32(uniforms.xShape.z));
          let sourceNearestRC = vec2<i32>(
            min(inputShapeRC - 1.0, floor(sourceFracIndexRC + uniforms.roundBase)));
          let newValue = getX(b, sourceNearestRC.x, sourceNearestRC.y, d);

          setOutputAtIndex(index, newValue);
        }
      }
    `}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sA(s){const{inputs:e,backend:t,attrs:n}=s,{images:r}=e,{alignCorners:i,halfPixelCenters:o,size:a}=n,[l,u]=a,c=i&&l>1?1:0,h=i&&u>1?1:0,w=[{type:"float32",data:[c,h]},{type:"float32",data:[i?.5:0]}],k=new nA(r.shape,l,u,o);return t.runWebGPUProgram(k,[r],r.dtype,w)}const rA={kernelName:rp,backendName:"webgpu",kernelFunc:sA};/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class iA{constructor(e){this.uniforms="",this.workPerThread=1,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=xo(e,1),this.variableNames=e.map((t,n)=>`T${n}`),this.dispatchLayout=fn(this.outputShape),this.dispatch=ft(this.dispatchLayout,this.outputShape,this.workgroupSize,[this.workPerThread,1,1]),this.offsetLength=e.length-1;for(let t=0;t<this.offsetLength;t++)this.uniforms+=`offset${t} : i32,`;this.shaderKey="concat"}getUserCode(){const e=[];if(this.offsetLength>0){e.push("if (yC < uniforms.offset0){ setOutputAtCoords(coords.x, coords.y, getT0(yR, yC)); }");for(let i=1;i<this.offsetLength;i++)e.push(`else if (yC < uniforms.offset${[i]}){ setOutputAtCoords(coords.x, coords.y, getT${i}(yR, yC - uniforms.offset${i-1})); }`);const n=this.offsetLength,r=this.offsetLength-1;e.push(`else { setOutputAtCoords(coords.x, coords.y, getT${n}(yR, yC - uniforms.offset${r})); }`)}else e.push("setOutputAtCoords(coords.x, coords.y, getT0(yR, yC));");return`
      ${st("index")} {
        for(var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let flatIndex = index * ${this.workPerThread} + i;
          if(flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndex);
            let yR = coords.x;
            let yC = coords.y;

            ${e.join(`
        `)}
          }
        }
      }
    `}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function oA(s){const{inputs:e,backend:t}=s,{real:n,imag:r}=e,i=t.makeTensorInfo(n.shape,"complex64"),o=t.tensorMap.get(i.dataId),a=Ls({inputs:{x:n},backend:t}),l=Ls({inputs:{x:r},backend:t});return o.complexTensorInfos={real:a,imag:l},i}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function aA(s){const{inputs:e,backend:t}=s,{input:n}=e,r=t.tensorMap.get(n.dataId);return Ls({inputs:{x:r.complexTensorInfos.imag},backend:t})}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lA(s){const{inputs:e,backend:t}=s,{input:n}=e,r=t.tensorMap.get(n.dataId);return Ls({inputs:{x:r.complexTensorInfos.real},backend:t})}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gi(s,e,t){const n=s[0].dtype;if(n==="complex64"){const k=s.map(f=>lA({inputs:{input:f},backend:t})),A=s.map(f=>aA({inputs:{input:f},backend:t})),m=Gi(k,e,t),S=Gi(A,e,t),b=oA({inputs:{real:m,imag:S},backend:t});return k.forEach(f=>t.disposeData(f.dataId)),A.forEach(f=>t.disposeData(f.dataId)),t.disposeData(m.dataId),t.disposeData(S.dataId),b}let r=t.shouldExecuteOnCPU(s);if(n==="string"&&(r=!0),r){const k=s.map(_=>{const D=[-1,me(_.shape.slice(e))];return Ve({inputs:{x:_},backend:t,attrs:{shape:D}})}),A=k.map(_=>({vals:t.readSync(_.dataId),shape:_.shape})),m=xo(k.map(_=>_.shape),1),S=k[0].shape[0]===1,b=HI(A,m,n,S),f=xo(s.map(_=>_.shape),e),v=t.makeTensorInfo(f,n,b);return k.forEach(_=>t.disposeData(_.dataId)),v}const i=t.device.limits.maxStorageBuffersPerShaderStage-1;if(s.length>i){const k=[];for(let m=0;m<s.length;m+=i){const S=s.slice(m,m+i);k.push(Gi(S,e,t))}const A=Gi(k,e,t);for(const m of k)t.disposeData(m.dataId);return A}const{tensors2D:o,outShape:a}=uA(s,e,t),l=o.map(k=>k.shape),u=new iA(l),c=[],h=new Array(l.length-1);if(h.length>0){h[0]=l[0][1],c.push({type:"int32",data:[h[0]]});for(let k=1;k<h.length;k++)h[k]=h[k-1]+l[k][1],c.push({type:"int32",data:[h[k]]})}const d=t.runWebGPUProgram(u,o,o[0].dtype,c);o.forEach(k=>t.disposeData(k.dataId));const w=Ve({inputs:{x:d},backend:t,attrs:{shape:a}});return t.disposeData(d.dataId),w}function uA(s,e,t){const n=xo(s.map(i=>i.shape),e);return{tensors2D:s.map(i=>Ve({inputs:{x:i},backend:t,attrs:{shape:[me(i.shape.slice(0,e)),me(i.shape.slice(e))]}})),outShape:n}}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cA(s){const{inputs:e,backend:t,attrs:n}=s,{axis:r}=n,i=Co(r,e[0].shape)[0],o=e.map(u=>u.shape);tv(o,i);const a=xo(e.map(u=>u.shape),i);if(me(a)===0)return t.makeTensorInfo(a,e[0].dtype,[]);const l=e.filter(u=>me(u.shape)>0);return l.length===1?Ls({inputs:{x:l[0]},backend:t}):Gi(l,i,t)}const hA={kernelName:tp,backendName:"webgpu",kernelFunc:cA},fA=[Mk,Uk,eE,WT,tA,rA,hA,Pk];for(const s of fA)Zb({...s,backendName:"webgpu-oidn"});async function dA(){try{const s={powerPreference:"high-performance"},e=await navigator.gpu.requestAdapter(s),t={},n=[];e.features.has("timestamp-query")&&n.push("timestamp-query"),e.features.has("bgra8unorm-storage")&&n.push(["bgra8unorm-storage"]),t.requiredFeatures=n;const r=e.limits;t.requiredLimits={maxComputeWorkgroupStorageSize:r.maxComputeWorkgroupStorageSize,maxComputeWorkgroupsPerDimension:r.maxComputeWorkgroupsPerDimension,maxStorageBufferBindingSize:r.maxStorageBufferBindingSize,maxBufferSize:r.maxBufferSize,maxComputeWorkgroupSizeX:r.maxComputeWorkgroupSizeX,maxComputeInvocationsPerWorkgroup:r.maxComputeInvocationsPerWorkgroup};const i=await e.requestDevice(t),o=e.info??await e.requestAdapterInfo?.();return pA(i,o)}catch{}}async function pA(s,e){let t=K.findBackend("webgpu-oidn");return t!=null||(t=new zo(s,e),K.registerBackend("webgpu-oidn",()=>t),await K.setBackend("webgpu-oidn")),t}async function mA(s,e,t){const n=await dA(),r=l0(s);return new ck(r,n,t)}async function gA(s,e,t){return fetch(s).then(n=>n.arrayBuffer()).then(n=>mA(n,e,t))}var yA=`const EPS = 1e-4;\r
const INF = 1e20f;\r
alias Refl_t = u32;\r
\r
const DIFF: Refl_t = 0;\r
const SPEC: Refl_t = 1;\r
const REFR: Refl_t = 2;\r
\r
var<private> rnd: vec3u;\r
const M_PI = radians(180);\r
const M_1_PI = 1.0 / M_PI;\r
\r
override SAMPLES: f32 = 1.0;\r
const GAMMA = vec3f(1 / 2.2);\r
override DIMENSION_SIZE = 16u;\r
\r
struct Sphere\r
{\r
    p: vec3f, rad: f32,\r
    e: vec3f, refl: Refl_t, c: vec3f\r
};\r
\r
struct Ray { o: vec3f, d: vec3f };\r
\r
fn init_rnd(id: vec3u, seed: vec3u)\r
{\r
    const A = vec3(\r
        1741651 * 1009,\r
        140893 * 1609 * 13,\r
        6521 * 983 * 7 * 2\r
    );\r
\r
    rnd = (id * A) ^ seed;\r
}\r
\r
fn rand() -> f32\r
{\r
    const C = vec3(\r
        60493 * 9377,\r
        11279 * 2539 * 23,\r
        7919 * 631 * 5 * 3\r
    );\r
\r
    rnd = (rnd * C) ^ (rnd.yzx >> vec3(4u));\r
    return f32(rnd.x ^ rnd.y) / f32(0xffffffff);\r
}\r
\r
fn intersect_sphere(s: Sphere, r: Ray) -> f32\r
{\r
    let op = s.p - r.o;\r
    let b = dot(op, r.d);\r
\r
    var det = b * b - dot(op, op) + s.rad * s.rad;\r
\r
    if (det < 0) { return 0; }\r
    else { det = sqrt(det); }\r
\r
    var t = b - det;\r
\r
    if (t > EPS) { return t; }\r
    else\r
    {\r
        t = b + det;\r
        return select(0, t, t > EPS);\r
    }\r
}\r
\r
fn intersect(r: Ray, t: ptr<function, f32>, id: ptr<function, i32>) -> bool\r
{\r
    *t = INF;\r
\r
    for (var s = i32(SPHERES - 1); s > -1; s--)\r
    {\r
        let d = intersect_sphere(spheres[s], r);\r
\r
        if (d != 0f && d < *t)\r
        {\r
            *t = d;\r
            *id = s;\r
        }\r
    }\r
\r
    return *t < INF;\r
}\r
\r
fn radiance(ray: Ray, depth: u32) -> vec3f\r
{\r
    var E = 1;\r
    var t: f32;\r
    var id = 0;\r
\r
    var r = ray;\r
    var d = depth;\r
\r
    var e = vec3f(0);\r
    var cl = vec3f(0);\r
    var cf = vec3f(1);\r
\r
    loop\r
    {\r
        if (!intersect(r, &t, &id)) { return cl; }\r
\r
        let obj = spheres[id];\r
\r
        let x = r.o + r.d * t;\r
        let n = normalize(x - obj.p);\r
        let nl = select(-n, n, dot(n, r.d) < 0);\r
        var f = obj.c;\r
\r
        let p = max(max(f.x, f.y), f.z);\r
        cl += cf * (obj.e * f32(E) + e);\r
\r
        d++;\r
        if (d > 5 || p == 0)\r
        {\r
            if (rand() < p) { f *= (1 / p); }\r
            else { return cl; }\r
        }\r
\r
        cf *= f;\r
\r
        if (obj.refl == DIFF)\r
        {\r
            let r1 = 2 * M_PI * rand();\r
            let r2 = rand();\r
            let r2s = sqrt(r2);\r
\r
            let w = nl;\r
            let u = normalize(cross(select(vec3f(1, 0, 0), vec3f(0, 1, 0), abs(w.x) > 0.1), w));\r
            let v = cross(w, u);\r
\r
            let d = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2));\r
\r
            e = vec3f(0);\r
            for (var i = 0; i < i32(SPHERES); i++)\r
            {\r
                let s = spheres[i];\r
\r
                if (s.e.x <= 0 && s.e.y <= 0 && s.e.z <= 0) { continue; }\r
\r
                let sw = s.p - x;\r
                // \`abs(w.x) > 0.1\` was replaced by \`abs(w.x) > EPS\` to fix the vertical line artifact:\r
                let su = normalize(cross(select(vec3f(1, 0, 0), vec3f(0, 1, 0), abs(sw.x) > EPS), sw));\r
                let sv = cross(sw, su);\r
\r
                let cos_a_max = sqrt(1 - s.rad * s.rad / dot(x - s.p, x - s.p));\r
\r
                let eps1 = rand();\r
                let eps2 = rand();\r
\r
                let cos_a = 1 - eps1 + eps1 * cos_a_max;\r
                let sin_a = sqrt(1 - cos_a * cos_a);\r
                let phi = 2 * M_PI * eps2;\r
\r
                let l = normalize(su * cos(phi) * sin_a + sv * sin(phi) * sin_a + sw * cos_a);\r
\r
                if (intersect(Ray(x, l), &t, &id) && id == i)\r
                {\r
                    let omega = 2 * M_PI * (1 - cos_a_max);\r
                    e += f * (s.e * dot(l, nl) * omega) * M_1_PI;\r
                }\r
            }\r
\r
            r = Ray(x, d);\r
            E = 0;\r
            continue;\r
        }\r
        else if (obj.refl == SPEC)\r
        {\r
            r = Ray(x, r.d - n * 2 * dot(n, r.d));\r
            continue;\r
        }\r
\r
        let reflRay = Ray(x, r.d - n * 2 * dot(n, r.d));\r
        let into = dot(n, nl) > 0;\r
\r
        let nc = 1f;\r
        let nt = 1.5;\r
        let nnt = select(nt / nc, nc / nt, into);\r
        let ddn = dot(r.d, nl);\r
\r
        let cos2t = 1 - nnt * nnt * (1 - ddn * ddn);\r
\r
        if (cos2t < 0)\r
        {\r
            r = reflRay;\r
            continue;\r
        }\r
\r
        let tdir = normalize(r.d * nnt - n * select(-1f, 1f, into) * (ddn * nnt + sqrt(cos2t)));\r
\r
        let a = nt - nc;\r
        let b = nt + nc;\r
        let R0 = a * a / (b * b);\r
        let c = 1 - select(dot(tdir, n), -ddn, into);\r
\r
        let Re = R0 + (1 - R0) * c * c * c * c * c;\r
        let Tr = 1 - Re;\r
        let P = 0.25 + 0.5 * Re;\r
        let RP = Re / P;\r
        let TP = Tr / (1 - P);\r
\r
        if (rand() < P)\r
        {\r
            cf *= RP;\r
            r = reflRay;\r
        }\r
        else\r
        {\r
            cf *= TP;\r
            r = Ray(x, tdir);\r
        }\r
\r
        continue;\r
    }\r
}\r
\r
@group(0) @binding(0) var<uniform> res: vec3f;\r
@group(0) @binding(1) var<uniform> seed: vec3u;\r
@group(0) @binding(2) var<storage, read_write> color3f: array<vec3f>;\r
@group(0) @binding(3) var<storage, read_write> color4u: array<vec4u>;\r
@group(0) @binding(4) var<storage, read> spheres: array<Sphere, SPHERES>;\r
\r
@compute @workgroup_size(DIMENSION_SIZE, DIMENSION_SIZE)\r
fn compute(@builtin(global_invocation_id) globalInvocation: vec3u)\r
{\r
    let coord = vec2f(globalInvocation.xy);\r
    init_rnd(globalInvocation, seed);\r
\r
    if (all(coord < res.xy))\r
    {\r
        let cam = Ray(\r
            vec3f(50, 52, 295.6),\r
            normalize(vec3f(0, -0.042612, -1))\r
        );\r
\r
        let iy = res.y - coord.y - 1;\r
        let i = u32(coord.x + iy * res.x);\r
\r
        let cx = vec3f(res.x * 0.5135 / res.y, 0, 0);\r
        let cy = normalize(cross(cx, cam.d)) * 0.5135;\r
\r
        for (var sy = 0u; sy < 2; sy++)\r
        {\r
            for (var sx = 0u; sx < 2; sx++)\r
            {\r
                let r1 = rand() * 2;\r
                let r2 = rand() * 2;\r
\r
                let dx = select(1 - sqrt(2 - r1), sqrt(r1) - 1, r1 < 1);\r
                let dy = select(1 - sqrt(2 - r2), sqrt(r2) - 1, r2 < 1);\r
\r
                var d = cx * (((f32(sx) + 0.5 + dx) / 2 + coord.x) / res.x - 0.5) +\r
                        cy * (((f32(sy) + 0.5 + dy) / 2 + coord.y) / res.y - 0.5) + cam.d;\r
\r
                let r = radiance(Ray(cam.o + d * 140, normalize(d)), 0) * SAMPLES;\r
                color3f[i] = color3f[i] + clamp(r, vec3f(0), vec3f(1)) * 0.25;\r
            }\r
        }\r
\r
        color4u[i] = vec4u(vec3u(\r
            pow(clamp(color3f[i], vec3f(0), vec3f(1)), GAMMA) * 255 + 0.5\r
        ), 255);\r
    }\r
}\r
`,bA=`struct VertexOutput\r
{\r
    @builtin(position) position: vec4f,\r
    @location(1) @interpolate(flat) res: vec2u\r
};\r
\r
@group(0) @binding(1) var<storage, read_write> color4u: array<vec4u>;\r
\r
@vertex fn vertex(@builtin(vertex_index) index: u32) -> VertexOutput\r
{\r
    var output: VertexOutput;\r
    let position = GetQuadCoord(index);\r
\r
    output.position = vec4f(position, 0, 1);\r
    output.res = vec2u(resolution.xy);\r
\r
    return output;\r
}\r
\r
@fragment fn fragment(\r
    @builtin(position) position: vec4f,\r
    @location(1) @interpolate(flat) res: vec2u\r
) -> @location(0) vec4f\r
{\r
    let pos = vec2u(position.xy);\r
\r
    let x = pos.x % res.x;\r
    let y = pos.y % res.y;\r
    let i = x + y * res.x;\r
\r
    return vec4f(color4u[i]) / 255;\r
}\r
`,wA=Object.defineProperty,py=s=>{throw TypeError(s)},xA=(s,e,t)=>e in s?wA(s,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):s[e]=t,Lt=(s,e,t)=>xA(s,typeof e!="symbol"?e+"":e,t),yh=(s,e,t)=>e.has(s)||py("Cannot "+t),T=(s,e,t)=>(yh(s,e,"read from private field"),t?t.call(s):e.get(s)),ee=(s,e,t)=>e.has(s)?py("Cannot add the same private member more than once"):e instanceof WeakSet?e.add(s):e.set(s,t),j=(s,e,t,n)=>(yh(s,e,"write to private field"),e.set(s,t),t),se=(s,e,t)=>(yh(s,e,"access private method"),t),rs,is,os,nn;const _A=class my{constructor(e=0,t,n,r=255){ee(this,rs,0),ee(this,is,0),ee(this,os,0),ee(this,nn,1),typeof t=="number"&&typeof n=="number"?this.RGBA=[e,t,n,r]:(j(this,rs,(e>>16&255)/255),j(this,is,(e>>8&255)/255),j(this,os,(255&e)/255),j(this,nn,r/255))}Set(e,t=255){return j(this,rs,(e>>16&255)/255),j(this,is,(e>>8&255)/255),j(this,os,(255&e)/255),j(this,nn,t/255),this}Premultiply(e,t){t??(t=new my),e??(e=T(this,nn));const n=T(this,rs)*e,r=T(this,is)*e,i=T(this,os)*e;return t.rgba=[n,r,i,e],t}set rgb(e){j(this,rs,e[0]),j(this,is,e[1]),j(this,os,e[2]),j(this,nn,e[3]??1)}get rgb(){return[T(this,rs),T(this,is),T(this,os)]}set a(e){j(this,nn,e)}get a(){return T(this,nn)}set rgba(e){this.rgb=e}get rgba(){return this.rgb.concat(T(this,nn))}set RGB(e){j(this,rs,e[0]/255),j(this,is,e[1]/255),j(this,os,e[2]/255),j(this,nn,(e[3]??255)/255)}get RGB(){return[255*T(this,rs),255*T(this,is),255*T(this,os)]}set A(e){j(this,nn,e/255)}get A(){return 255*T(this,nn)}set RGBA(e){this.RGB=e}get RGBA(){return this.RGB.concat(this.A)}};rs=new WeakMap,is=new WeakMap,os=new WeakMap,nn=new WeakMap;let gl=_A;Ue({RAD:Math.PI/180,DEG:180/Math.PI,HPI:Math.PI/2,TAU:2*Math.PI});Ue({DEVICE_LOST:"Device::Lost"});const ie=Ue({FORMAT_NOT_SUPPORTED:"FORMAT_NOT_SUPPORTED",WEBGPU_NOT_SUPPORTED:"WEBGPU_NOT_SUPPORTED",ADAPTER_NOT_FOUND:"ADAPTER_NOT_FOUND",FEATURE_NOT_FOUND:"FEATURE_NOT_FOUND",DEVICE_NOT_FOUND:"DEVICE_NOT_FOUND",DEVICE_NOT_REQUESTED:"DEVICE_NOT_REQUESTED",DEVICE_LOST:"DEVICE_LOST",SHADER_CODE_NOT_FOUND:"SHADER_CODE_NOT_FOUND",SHADER_MODULE_NOT_FOUND:"SHADER_MODULE_NOT_FOUND",VERTEX_ENTRY_NOT_FOUND:"VERTEX_ENTRY_NOT_FOUND",VERTEX_ATTRIBUTE_NOT_FOUND:"VERTEX_ATTRIBUTE_NOT_FOUND",UNIFORM_NOT_FOUND:"UNIFORM_NOT_FOUND",STORAGE_NOT_FOUND:"STORAGE_NOT_FOUND",INVALID_UNIFORM_NAME:"INVALID_UNIFORM_NAME",BINDING_NOT_FOUND:"BINDING_NOT_FOUND",PIPELINE_NOT_FOUND:"PIPELINE_NOT_FOUND",LEGACY_RENDER_PIPELINE_NOT_FOUND:"LEGACY_RENDER_PIPELINE_NOT_FOUND",RENDERER_NOT_FOUND:"RENDERER_NOT_FOUND",TEXTURE_SIZE_NOT_FOUND:"TEXTURE_SIZE_NOT_FOUND",TEXTURE_NOT_FOUND:"TEXTURE_NOT_FOUND",INVALID_BYTES_PER_ROW:"INVALID_BYTES_PER_ROW",CANVAS_NOT_FOUND:"CANVAS_NOT_FOUND",CONTEXT_NOT_FOUND:"CONTEXT_NOT_FOUND",RENDER_PASS_NOT_FOUND:"RENDER_PASS_NOT_FOUND",COMMAND_ENCODER_NOT_FOUND:"COMMAND_ENCODER_NOT_FOUND",FONT_TEXTURE_NOT_FOUND:"FONT_TEXTURE_NOT_FOUND",TIMESTAMP_QUERY_NOT_FOUND:"TIMESTAMP_QUERY_NOT_FOUND",RENDER_PASS_ENDED:"RENDER_PASS_ENDED"}),gy=Ue({FORMAT_NOT_SUPPORTED:"Format is not yet supported: ",WEBGPU_NOT_SUPPORTED:"WebGPU is not supported in this browser.",ADAPTER_NOT_FOUND:"Failed to get a GPUAdapter.",DEVICE_NOT_FOUND:"Failed to get a GPUDevice.",FEATURE_NOT_FOUND:"Failed to get a GPUFeature ",DEVICE_NOT_REQUESTED:"GPUDevice was not requested.",DEVICE_LOST:"WebGPU device was lost. ",SHADER_CODE_NOT_FOUND:`Failed to get a WGSL shader when creating shader module.
        An empty shader will be used instead.`,SHADER_MODULE_NOT_FOUND:"Failed to get shader module in ",VERTEX_ENTRY_NOT_FOUND:"Failed to find function ",VERTEX_ATTRIBUTE_NOT_FOUND:"Failed to find vertex attribute ",UNIFORM_NOT_FOUND:"Failed to find uniform ",STORAGE_NOT_FOUND:"Failed to find storage ",INVALID_UNIFORM_NAME:"Requested uniform is already in use and managed internally: ",BINDING_NOT_FOUND:"Failed to find binding ",PIPELINE_NOT_FOUND:"Failed to get GPU",LEGACY_RENDER_PIPELINE_NOT_FOUND:'"Device.RenderPipeline" instance is required in `LegacyTexture` for this operation.\n        Pass it to the `LegacyTexture` constructor or use `Texture.LegacyRenderer` setter before ',RENDERER_NOT_FOUND:'"Device.Renderer" instance is required in `Texture` for this operation.\n        Pass it to the `Texture` constructor or use `Texture.Renderer` setter before ',TEXTURE_SIZE_NOT_FOUND:"`size` array or a `width` value is required in `options` parameter of ",TEXTURE_NOT_FOUND:"`options` is required to have a `texture` value or its `create` entry\n        to be either `true` or a `TextureDescriptor` object when calling ",INVALID_BYTES_PER_ROW:"`bytesPerRow` parameter is not a multiple of 256 in ",CANVAS_NOT_FOUND:"Failed to get a WebGPU canvas.",CONTEXT_NOT_FOUND:"Failed to get a WebGPU context.",RENDER_PASS_NOT_FOUND:"Failed to use pipeline in render pass because it has not started.",COMMAND_ENCODER_NOT_FOUND:"Failed to get a GPUCommandEncoder.",FONT_TEXTURE_NOT_FOUND:"Failed to find font texture in ",TIMESTAMP_QUERY_NOT_FOUND:'"timestamp-query" feature is required to be set with\n        `Device.SetRequiredFeatures` when creating a new `GPUTiming` instance.',RENDER_PASS_ENDED:"Failed get a render pass because it has ended.\n        `Render` method has to be called with `submit` flag set to `false`."}),vA=Ue({WEBGPU_NOT_SUPPORTED:0,ADAPTER_NOT_FOUND:1,DEVICE_NOT_FOUND:2,DEVICE_NOT_REQUESTED:3,DEVICE_LOST:4,CANVAS_NOT_FOUND:5,CONTEXT_NOT_FOUND:6,COMMAND_ENCODER_NOT_FOUND:7,PIPELINE_NOT_FOUND:8});function Bt(s,e){console.warn(`${gy[s]}${e??""}`.replace(/\s\s+/g," "))}function ue(s,e){throw Error(`${gy[s]}${e??""}`.replace(/\s\s+/g," "),{cause:vA[s]})}const Ct=Ue({INDEX:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST,VERTEX:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST,STORAGE:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,UNIFORM:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,READABLE:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,WRITABLE:GPUBufferUsage.MAP_WRITE|GPUBufferUsage.COPY_SRC,QUERY:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),cd={operation:"add",srcFactor:"one",dstFactor:"zero"},hd={operation:"add",srcFactor:"one",dstFactor:"one"},fd={operation:"add",srcFactor:"one",dstFactor:"one-minus-src-alpha"},dd={operation:"add",srcFactor:"one-minus-dst-alpha",dstFactor:"one"},pd={operation:"add",srcFactor:"dst-alpha",dstFactor:"zero"},md={operation:"add",srcFactor:"zero",dstFactor:"src-alpha"},gd={operation:"add",srcFactor:"one-minus-dst-alpha",dstFactor:"zero"},yd={operation:"add",srcFactor:"zero",dstFactor:"one-minus-src-alpha"},bd={operation:"add",srcFactor:"dst-alpha",dstFactor:"one-minus-src-alpha"},wd={operation:"add",srcFactor:"one-minus-dst-alpha",dstFactor:"src-alpha"};Ue({COPY:Ue({color:cd,alpha:cd}),ADDITIVE:Ue({color:hd,alpha:hd}),SOURCE_OVER:Ue({color:fd,alpha:fd}),DESTINATION_OVER:Ue({color:dd,alpha:dd}),SOURCE_IN:Ue({color:pd,alpha:pd}),DESTINATION_IN:Ue({color:md,alpha:md}),SOURCE_OUT:Ue({color:gd,alpha:gd}),DESTINATION_OUT:Ue({color:yd,alpha:yd}),SOURCE_ATOP:Ue({color:bd,alpha:bd}),DESTINATION_ATOP:Ue({color:wd,alpha:wd})});class an{constructor(e,t){this.name=e,this.attributes=t,this.size=0}get isArray(){return!1}get isStruct(){return!1}get isTemplate(){return!1}get isPointer(){return!1}getTypeName(){return this.name}}class xd{constructor(e,t,n){this.name=e,this.type=t,this.attributes=n,this.offset=0,this.size=0}get isArray(){return this.type.isArray}get isStruct(){return this.type.isStruct}get isTemplate(){return this.type.isTemplate}get align(){return this.type.isStruct?this.type.align:0}get members(){return this.type.isStruct?this.type.members:null}get format(){return this.type.isArray||this.type.isTemplate?this.type.format:null}get count(){return this.type.isArray?this.type.count:0}get stride(){return this.type.isArray?this.type.stride:this.size}}class Ns extends an{constructor(e,t){super(e,t),this.members=[],this.align=0,this.startLine=-1,this.endLine=-1,this.inUse=!1}get isStruct(){return!0}}class Ps extends an{constructor(e,t){super(e,t),this.count=0,this.stride=0}get isArray(){return!0}getTypeName(){return`array<${this.format.getTypeName()}, ${this.count}>`}}class tc extends an{constructor(e,t,n){super(e,n),this.format=t}get isPointer(){return!0}getTypeName(){return"&"+this.format.getTypeName()}}class Ar extends an{constructor(e,t,n,r){super(e,n),this.format=t,this.access=r}get isTemplate(){return!0}getTypeName(){let e=this.name;if(this.format!==null){if(e==="vec2"||e==="vec3"||e==="vec4"||e==="mat2x2"||e==="mat2x3"||e==="mat2x4"||e==="mat3x2"||e==="mat3x3"||e==="mat3x4"||e==="mat4x2"||e==="mat4x3"||e==="mat4x4"){if(this.format.name==="f32")return e+="f",e;if(this.format.name==="i32")return e+="i",e;if(this.format.name==="u32")return e+="u",e;if(this.format.name==="bool")return e+="b",e;if(this.format.name==="f16")return e+="h",e}e+=`<${this.format.name}>`}else if(e==="vec2"||e==="vec3"||e==="vec4")return e;return e}}var Is;(s=>{s[s.Uniform=0]="Uniform",s[s.Storage=1]="Storage",s[s.Texture=2]="Texture",s[s.Sampler=3]="Sampler",s[s.StorageTexture=4]="StorageTexture"})(Is||(Is={}));class ua{constructor(e,t,n,r,i,o,a){this.name=e,this.type=t,this.group=n,this.binding=r,this.attributes=i,this.resourceType=o,this.access=a}get isArray(){return this.type.isArray}get isStruct(){return this.type.isStruct}get isTemplate(){return this.type.isTemplate}get size(){return this.type.size}get align(){return this.type.isStruct?this.type.align:0}get members(){return this.type.isStruct?this.type.members:null}get format(){return this.type.isArray||this.type.isTemplate?this.type.format:null}get count(){return this.type.isArray?this.type.count:0}get stride(){return this.type.isArray?this.type.stride:this.size}}class SA{constructor(e,t){this.name=e,this.type=t}}class kA{constructor(e,t,n,r){this.name=e,this.type=t,this.locationType=n,this.location=r,this.interpolation=null}}class _d{constructor(e,t,n,r){this.name=e,this.type=t,this.locationType=n,this.location=r}}class IA{constructor(e,t,n,r){this.name=e,this.type=t,this.attributes=n,this.id=r}}class EA{constructor(e,t,n){this.name=e,this.type=t,this.attributes=n}}class TA{constructor(e,t=null,n){this.stage=null,this.inputs=[],this.outputs=[],this.arguments=[],this.returnType=null,this.resources=[],this.overrides=[],this.startLine=-1,this.endLine=-1,this.inUse=!1,this.calls=new Set,this.name=e,this.stage=t,this.attributes=n}}class AA{constructor(){this.vertex=[],this.fragment=[],this.compute=[]}}const yy=new Float32Array(1),CA=new Int32Array(yy.buffer),xt=new Uint16Array(1);function NA(s){yy[0]=s;const e=CA[0],t=e>>31&1;let n=e>>23&255,r=8388607&e;if(n===255)return xt[0]=t<<15|31744|(r!==0?512:0),xt[0];if(n===0){if(r===0)return xt[0]=t<<15,xt[0];r|=8388608;let i=113;for(;!(8388608&r);)r<<=1,i--;return n=127-i,r&=8388607,n>0?(r=(r>>126-n)+(r>>127-n&1),xt[0]=t<<15|n<<10|r>>13,xt[0]):(xt[0]=t<<15,xt[0])}return n=n-127+15,n>=31?(xt[0]=t<<15|31744,xt[0]):n<=0?n<-10?(xt[0]=t<<15,xt[0]):(r=(8388608|r)>>1-n,xt[0]=t<<15|r>>13,xt[0]):(r>>=13,xt[0]=t<<15|n<<10|r,xt[0])}const bh=new Uint32Array(1),by=new Float32Array(bh.buffer,0,1);function vd(s){const e=112+(s>>6&31)<<23|(63&s)<<17;return bh[0]=e,by[0]}function ve(s,e,t,n){const r=[0,0,0,0];for(let u=0;u<n;++u)switch(t){case"8unorm":r[u]=s[e]/255,e++;break;case"8snorm":r[u]=s[e]/255*2-1,e++;break;case"8uint":r[u]=s[e],e++;break;case"8sint":r[u]=s[e]-127,e++;break;case"16uint":r[u]=s[e]|s[e+1]<<8,e+=2;break;case"16sint":r[u]=(s[e]|s[e+1]<<8)-32768,e+=2;break;case"16float":r[u]=(o=(32768&(i=s[e]|s[e+1]<<8))>>15,l=1023&i,(a=(31744&i)>>10)==0?(o?-1:1)*Math.pow(2,-14)*(l/1024):a==31?l?NaN:1/0*(o?-1:1):(o?-1:1)*Math.pow(2,a-15)*(1+l/1024)),e+=2;break;case"32uint":case"32sint":r[u]=s[e]|s[e+1]<<8|s[e+2]<<16|s[e+3]<<24,e+=4;break;case"32float":r[u]=new Float32Array(s.buffer,e,1)[0],e+=4}var i,o,a,l;return r}function Ee(s,e,t,n,r){for(let i=0;i<n;++i)switch(t){case"8unorm":s[e]=255*r[i],e++;break;case"8snorm":s[e]=127.5*(r[i]+1),e++;break;case"8uint":s[e]=r[i],e++;break;case"8sint":s[e]=r[i]+127,e++;break;case"16uint":new Uint16Array(s.buffer,e,1)[0]=r[i],e+=2;break;case"16sint":new Int16Array(s.buffer,e,1)[0]=r[i],e+=2;break;case"16float":{const o=NA(r[i]);new Uint16Array(s.buffer,e,1)[0]=o,e+=2;break}case"32uint":new Uint32Array(s.buffer,e,1)[0]=r[i],e+=4;break;case"32sint":new Int32Array(s.buffer,e,1)[0]=r[i],e+=4;break;case"32float":new Float32Array(s.buffer,e,1)[0]=r[i],e+=4}return r}const yu={r8unorm:{bytesPerBlock:1,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},r8snorm:{bytesPerBlock:1,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},r8uint:{bytesPerBlock:1,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},r8sint:{bytesPerBlock:1,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},rg8unorm:{bytesPerBlock:2,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rg8snorm:{bytesPerBlock:2,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rg8uint:{bytesPerBlock:2,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rg8sint:{bytesPerBlock:2,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rgba8unorm:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},"rgba8unorm-srgb":{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rgba8snorm:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rgba8uint:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rgba8sint:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},bgra8unorm:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},"bgra8unorm-srgb":{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},r16uint:{bytesPerBlock:2,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},r16sint:{bytesPerBlock:2,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},r16float:{bytesPerBlock:2,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},rg16uint:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rg16sint:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rg16float:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rgba16uint:{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rgba16sint:{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rgba16float:{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},r32uint:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},r32sint:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},r32float:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},rg32uint:{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rg32sint:{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rg32float:{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rgba32uint:{bytesPerBlock:16,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rgba32sint:{bytesPerBlock:16,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rgba32float:{bytesPerBlock:16,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rgb10a2uint:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rgb10a2unorm:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rg11b10ufloat:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},stencil8:{bytesPerBlock:1,blockWidth:1,blockHeight:1,isCompressed:!1,isDepthStencil:!0,hasDepth:!1,hasStencil:!0,channels:1},depth16unorm:{bytesPerBlock:2,blockWidth:1,blockHeight:1,isCompressed:!1,isDepthStencil:!0,hasDepth:!0,hasStencil:!1,channels:1},depth24plus:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,isDepthStencil:!0,hasDepth:!0,hasStencil:!1,depthOnlyFormat:"depth32float",channels:1},"depth24plus-stencil8":{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!1,isDepthStencil:!0,hasDepth:!0,hasStencil:!0,depthOnlyFormat:"depth32float",channels:1},depth32float:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,isDepthStencil:!0,hasDepth:!0,hasStencil:!1,channels:1},"depth32float-stencil8":{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!1,isDepthStencil:!0,hasDepth:!0,hasStencil:!0,stencilOnlyFormat:"depth32float",channels:1},rgb9e5ufloat:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},"bc1-rgba-unorm":{bytesPerBlock:8,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"bc1-rgba-unorm-srgb":{bytesPerBlock:8,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"bc2-rgba-unorm":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"bc2-rgba-unorm-srgb":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"bc3-rgba-unorm":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"bc3-rgba-unorm-srgb":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"bc4-r-unorm":{bytesPerBlock:8,blockWidth:4,blockHeight:4,isCompressed:!0,channels:1},"bc4-r-snorm":{bytesPerBlock:8,blockWidth:4,blockHeight:4,isCompressed:!0,channels:1},"bc5-rg-unorm":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:2},"bc5-rg-snorm":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:2},"bc6h-rgb-ufloat":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"bc6h-rgb-float":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"bc7-rgba-unorm":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"bc7-rgba-unorm-srgb":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"etc2-rgb8unorm":{bytesPerBlock:8,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"etc2-rgb8unorm-srgb":{bytesPerBlock:8,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"etc2-rgb8a1unorm":{bytesPerBlock:8,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"etc2-rgb8a1unorm-srgb":{bytesPerBlock:8,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"etc2-rgba8unorm":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"etc2-rgba8unorm-srgb":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"eac-r11unorm":{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!0,channels:1},"eac-r11snorm":{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!0,channels:1},"eac-rg11unorm":{bytesPerBlock:16,blockWidth:1,blockHeight:1,isCompressed:!0,channels:2},"eac-rg11snorm":{bytesPerBlock:16,blockWidth:1,blockHeight:1,isCompressed:!0,channels:2},"astc-4x4-unorm":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"astc-4x4-unorm-srgb":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"astc-5x4-unorm":{bytesPerBlock:16,blockWidth:5,blockHeight:4,isCompressed:!0,channels:4},"astc-5x4-unorm-srgb":{bytesPerBlock:16,blockWidth:5,blockHeight:4,isCompressed:!0,channels:4},"astc-5x5-unorm":{bytesPerBlock:16,blockWidth:5,blockHeight:5,isCompressed:!0,channels:4},"astc-5x5-unorm-srgb":{bytesPerBlock:16,blockWidth:5,blockHeight:5,isCompressed:!0,channels:4},"astc-6x5-unorm":{bytesPerBlock:16,blockWidth:6,blockHeight:5,isCompressed:!0,channels:4},"astc-6x5-unorm-srgb":{bytesPerBlock:16,blockWidth:6,blockHeight:5,isCompressed:!0,channels:4},"astc-6x6-unorm":{bytesPerBlock:16,blockWidth:6,blockHeight:6,isCompressed:!0,channels:4},"astc-6x6-unorm-srgb":{bytesPerBlock:16,blockWidth:6,blockHeight:6,isCompressed:!0,channels:4},"astc-8x5-unorm":{bytesPerBlock:16,blockWidth:8,blockHeight:5,isCompressed:!0,channels:4},"astc-8x5-unorm-srgb":{bytesPerBlock:16,blockWidth:8,blockHeight:5,isCompressed:!0,channels:4},"astc-8x6-unorm":{bytesPerBlock:16,blockWidth:8,blockHeight:6,isCompressed:!0,channels:4},"astc-8x6-unorm-srgb":{bytesPerBlock:16,blockWidth:8,blockHeight:6,isCompressed:!0,channels:4},"astc-8x8-unorm":{bytesPerBlock:16,blockWidth:8,blockHeight:8,isCompressed:!0,channels:4},"astc-8x8-unorm-srgb":{bytesPerBlock:16,blockWidth:8,blockHeight:8,isCompressed:!0,channels:4},"astc-10x5-unorm":{bytesPerBlock:16,blockWidth:10,blockHeight:5,isCompressed:!0,channels:4},"astc-10x5-unorm-srgb":{bytesPerBlock:16,blockWidth:10,blockHeight:5,isCompressed:!0,channels:4},"astc-10x6-unorm":{bytesPerBlock:16,blockWidth:10,blockHeight:6,isCompressed:!0,channels:4},"astc-10x6-unorm-srgb":{bytesPerBlock:16,blockWidth:10,blockHeight:6,isCompressed:!0,channels:4},"astc-10x8-unorm":{bytesPerBlock:16,blockWidth:10,blockHeight:8,isCompressed:!0,channels:4},"astc-10x8-unorm-srgb":{bytesPerBlock:16,blockWidth:10,blockHeight:8,isCompressed:!0,channels:4},"astc-10x10-unorm":{bytesPerBlock:16,blockWidth:10,blockHeight:10,isCompressed:!0,channels:4},"astc-10x10-unorm-srgb":{bytesPerBlock:16,blockWidth:10,blockHeight:10,isCompressed:!0,channels:4},"astc-12x10-unorm":{bytesPerBlock:16,blockWidth:12,blockHeight:10,isCompressed:!0,channels:4},"astc-12x10-unorm-srgb":{bytesPerBlock:16,blockWidth:12,blockHeight:10,isCompressed:!0,channels:4},"astc-12x12-unorm":{bytesPerBlock:16,blockWidth:12,blockHeight:12,isCompressed:!0,channels:4},"astc-12x12-unorm-srgb":{bytesPerBlock:16,blockWidth:12,blockHeight:12,isCompressed:!0,channels:4}};class dn{constructor(){this.id=dn._id++,this.line=0}get isAstNode(){return!0}get astNodeType(){return""}search(e){e(this)}searchBlock(e,t){if(e){t(yl.instance);for(const n of e)n instanceof Array?this.searchBlock(n,t):n.search(t);t(bl.instance)}}constEvaluate(e,t){throw Error("Cannot evaluate node")}constEvaluateString(e){return this.constEvaluate(e).toString()}}dn._id=0;class yl extends dn{}yl.instance=new yl;class bl extends dn{}bl.instance=new bl;const wy=new Set(["all","all","any","select","arrayLength","abs","acos","acosh","asin","asinh","atan","atanh","atan2","ceil","clamp","cos","cosh","countLeadingZeros","countOneBits","countTrailingZeros","cross","degrees","determinant","distance","dot","dot4U8Packed","dot4I8Packed","exp","exp2","extractBits","faceForward","firstLeadingBit","firstTrailingBit","floor","fma","fract","frexp","insertBits","inverseSqrt","ldexp","length","log","log2","max","min","mix","modf","normalize","pow","quantizeToF16","radians","reflect","refract","reverseBits","round","saturate","sign","sin","sinh","smoothStep","sqrt","step","tan","tanh","transpose","trunc","dpdx","dpdxCoarse","dpdxFine","dpdy","dpdyCoarse","dpdyFine","fwidth","fwidthCoarse","fwidthFine","textureDimensions","textureGather","textureGatherCompare","textureLoad","textureNumLayers","textureNumLevels","textureNumSamples","textureSample","textureSampleBias","textureSampleCompare","textureSampleCompareLevel","textureSampleGrad","textureSampleLevel","textureSampleBaseClampToEdge","textureStore","atomicLoad","atomicStore","atomicAdd","atomicSub","atomicMax","atomicMin","atomicAnd","atomicOr","atomicXor","atomicExchange","atomicCompareExchangeWeak","pack4x8snorm","pack4x8unorm","pack4xI8","pack4xU8","pack4x8Clamp","pack4xU8Clamp","pack2x16snorm","pack2x16unorm","pack2x16float","unpack4x8snorm","unpack4x8unorm","unpack4xI8","unpack4xU8","unpack2x16snorm","unpack2x16unorm","unpack2x16float","storageBarrier","textureBarrier","workgroupBarrier","workgroupUniformLoad","subgroupAdd","subgroupExclusiveAdd","subgroupInclusiveAdd","subgroupAll","subgroupAnd","subgroupAny","subgroupBallot","subgroupBroadcast","subgroupBroadcastFirst","subgroupElect","subgroupMax","subgroupMin","subgroupMul","subgroupExclusiveMul","subgroupInclusiveMul","subgroupOr","subgroupShuffle","subgroupShuffleDown","subgroupShuffleUp","subgroupShuffleXor","subgroupXor","quadBroadcast","quadSwapDiagonal","quadSwapX","quadSwapY"]);class ze extends dn{constructor(){super()}}class Eo extends ze{constructor(e,t,n,r,i,o){super(),this.calls=new Set,this.name=e,this.args=t,this.returnType=n,this.body=r,this.startLine=i,this.endLine=o}get astNodeType(){return"function"}search(e){if(this.attributes)for(const t of this.attributes)e(t);e(this);for(const t of this.args)e(t);this.searchBlock(this.body,e)}}class $A extends ze{constructor(e){super(),this.expression=e}get astNodeType(){return"staticAssert"}search(e){this.expression.search(e)}}class xy extends ze{constructor(e,t){super(),this.condition=e,this.body=t}get astNodeType(){return"while"}search(e){this.condition.search(e),this.searchBlock(this.body,e)}}class nc extends ze{constructor(e,t){super(),this.body=e,this.loopId=t}get astNodeType(){return"continuing"}search(e){this.searchBlock(this.body,e)}}class _y extends ze{constructor(e,t,n,r){super(),this.init=e,this.condition=t,this.increment=n,this.body=r}get astNodeType(){return"for"}search(e){var t,n,r;(t=this.init)===null||t===void 0||t.search(e),(n=this.condition)===null||n===void 0||n.search(e),(r=this.increment)===null||r===void 0||r.search(e),this.searchBlock(this.body,e)}}class ms extends ze{constructor(e,t,n,r,i){super(),this.attributes=null,this.name=e,this.type=t,this.storage=n,this.access=r,this.value=i}get astNodeType(){return"var"}search(e){var t;e(this),(t=this.value)===null||t===void 0||t.search(e)}}class wh extends ze{constructor(e,t,n){super(),this.attributes=null,this.name=e,this.type=t,this.value=n}get astNodeType(){return"override"}search(e){var t;(t=this.value)===null||t===void 0||t.search(e)}}class oo extends ze{constructor(e,t,n,r,i){super(),this.attributes=null,this.name=e,this.type=t,this.storage=n,this.access=r,this.value=i}get astNodeType(){return"let"}search(e){var t;e(this),(t=this.value)===null||t===void 0||t.search(e)}}class Ea extends ze{constructor(e,t,n,r,i){super(),this.attributes=null,this.name=e,this.type=t,this.storage=n,this.access=r,this.value=i}get astNodeType(){return"const"}constEvaluate(e,t){return this.value.constEvaluate(e,t)}search(e){var t;e(this),(t=this.value)===null||t===void 0||t.search(e)}}var Yr,Vi,G,F;(s=>{s.increment="++",s.decrement="--"})(Yr||(Yr={})),(s=>{s.parse=e=>{const t=e;if(t=="parse")throw Error("Invalid value for IncrementOperator");return s[t]}})(Yr||(Yr={}));class vy extends ze{constructor(e,t){super(),this.operator=e,this.variable=t}get astNodeType(){return"increment"}search(e){this.variable.search(e)}}(s=>{s.assign="=",s.addAssign="+=",s.subtractAssin="-=",s.multiplyAssign="*=",s.divideAssign="/=",s.moduloAssign="%=",s.andAssign="&=",s.orAssign="|=",s.xorAssign="^=",s.shiftLeftAssign="<<=",s.shiftRightAssign=">>="})(Vi||(Vi={})),(Vi||(Vi={})).parse=s=>{const e=s;if(e=="parse")throw Error("Invalid value for AssignOperator");return e};class Sy extends ze{constructor(e,t,n){super(),this.operator=e,this.variable=t,this.value=n}get astNodeType(){return"assign"}search(e){this.variable.search(e),this.value.search(e)}}class xh extends ze{constructor(e,t){super(),this.name=e,this.args=t}get astNodeType(){return"call"}isBuiltin(){return wy.has(this.name)}search(e){for(const t of this.args)t.search(e);e(this)}}class ky extends ze{constructor(e,t){super(),this.body=e,this.continuing=t}get astNodeType(){return"loop"}}class Iy extends ze{constructor(e,t){super(),this.condition=e,this.cases=t}get astNodeType(){return"switch"}search(e){e(this);for(const t of this.cases)t.search(e)}}class Ey extends ze{constructor(e,t,n,r){super(),this.condition=e,this.body=t,this.elseif=n,this.else=r}get astNodeType(){return"if"}search(e){this.condition.search(e),this.searchBlock(this.body,e),this.searchBlock(this.elseif,e),this.searchBlock(this.else,e)}}class Ty extends ze{constructor(e){super(),this.value=e}get astNodeType(){return"return"}search(e){var t;(t=this.value)===null||t===void 0||t.search(e)}}class DA extends ze{constructor(e){super(),this.name=e}get astNodeType(){return"enable"}}class OA extends ze{constructor(e){super(),this.extensions=e}get astNodeType(){return"requires"}}class Ay extends ze{constructor(e,t){super(),this.severity=e,this.rule=t}get astNodeType(){return"diagnostic"}}class _h extends ze{constructor(e,t){super(),this.name=e,this.type=t}get astNodeType(){return"alias"}}class MA extends ze{constructor(){super()}get astNodeType(){return"discard"}}class Cy extends ze{constructor(){super(),this.condition=null,this.loopId=-1}get astNodeType(){return"break"}}class Ny extends ze{constructor(){super(),this.loopId=-1}get astNodeType(){return"continue"}}class Y extends ze{constructor(e){super(),this.attributes=null,this.name=e}get astNodeType(){return"type"}get isStruct(){return!1}get isArray(){return!1}static maxFormatType(e){let t=e[0];if(t.name==="f32")return t;for(let n=1;n<e.length;++n){const r=Y._priority.get(t.name);Y._priority.get(e[n].name)<r&&(t=e[n])}return t.name==="x32"?Y.i32:t}getTypeName(){return this.name}}Y.x32=new Y("x32"),Y.f32=new Y("f32"),Y.i32=new Y("i32"),Y.u32=new Y("u32"),Y.f16=new Y("f16"),Y.bool=new Y("bool"),Y.void=new Y("void"),Y._priority=new Map([["f32",0],["f16",1],["u32",2],["i32",3],["x32",3]]);class Sd extends Y{constructor(e){super(e)}}class cs extends Y{constructor(e,t,n,r){super(e),this.members=t,this.startLine=n,this.endLine=r}get astNodeType(){return"struct"}get isStruct(){return!0}getMemberIndex(e){for(let t=0;t<this.members.length;t++)if(this.members[t].name==e)return t;return-1}search(e){for(const t of this.members)e(t)}}class z extends Y{constructor(e,t,n){super(e),this.format=t,this.access=n}get astNodeType(){return"template"}getTypeName(){let e=this.name;if(this.format!==null){if(e==="vec2"||e==="vec3"||e==="vec4"||e==="mat2x2"||e==="mat2x3"||e==="mat2x4"||e==="mat3x2"||e==="mat3x3"||e==="mat3x4"||e==="mat4x2"||e==="mat4x3"||e==="mat4x4"){if(this.format.name==="f32")return e+="f",e;if(this.format.name==="i32")return e+="i",e;if(this.format.name==="u32")return e+="u",e;if(this.format.name==="bool")return e+="b",e;if(this.format.name==="f16")return e+="h",e}e+=`<${this.format.name}>`}else if(e==="vec2"||e==="vec3"||e==="vec4")return e;return e}}z.vec2f=new z("vec2",Y.f32,null),z.vec3f=new z("vec3",Y.f32,null),z.vec4f=new z("vec4",Y.f32,null),z.vec2i=new z("vec2",Y.i32,null),z.vec3i=new z("vec3",Y.i32,null),z.vec4i=new z("vec4",Y.i32,null),z.vec2u=new z("vec2",Y.u32,null),z.vec3u=new z("vec3",Y.u32,null),z.vec4u=new z("vec4",Y.u32,null),z.vec2h=new z("vec2",Y.f16,null),z.vec3h=new z("vec3",Y.f16,null),z.vec4h=new z("vec4",Y.f16,null),z.vec2b=new z("vec2",Y.bool,null),z.vec3b=new z("vec3",Y.bool,null),z.vec4b=new z("vec4",Y.bool,null),z.mat2x2f=new z("mat2x2",Y.f32,null),z.mat2x3f=new z("mat2x3",Y.f32,null),z.mat2x4f=new z("mat2x4",Y.f32,null),z.mat3x2f=new z("mat3x2",Y.f32,null),z.mat3x3f=new z("mat3x3",Y.f32,null),z.mat3x4f=new z("mat3x4",Y.f32,null),z.mat4x2f=new z("mat4x2",Y.f32,null),z.mat4x3f=new z("mat4x3",Y.f32,null),z.mat4x4f=new z("mat4x4",Y.f32,null),z.mat2x2h=new z("mat2x2",Y.f16,null),z.mat2x3h=new z("mat2x3",Y.f16,null),z.mat2x4h=new z("mat2x4",Y.f16,null),z.mat3x2h=new z("mat3x2",Y.f16,null),z.mat3x3h=new z("mat3x3",Y.f16,null),z.mat3x4h=new z("mat3x4",Y.f16,null),z.mat4x2h=new z("mat4x2",Y.f16,null),z.mat4x3h=new z("mat4x3",Y.f16,null),z.mat4x4h=new z("mat4x4",Y.f16,null),z.mat2x2i=new z("mat2x2",Y.i32,null),z.mat2x3i=new z("mat2x3",Y.i32,null),z.mat2x4i=new z("mat2x4",Y.i32,null),z.mat3x2i=new z("mat3x2",Y.i32,null),z.mat3x3i=new z("mat3x3",Y.i32,null),z.mat3x4i=new z("mat3x4",Y.i32,null),z.mat4x2i=new z("mat4x2",Y.i32,null),z.mat4x3i=new z("mat4x3",Y.i32,null),z.mat4x4i=new z("mat4x4",Y.i32,null),z.mat2x2u=new z("mat2x2",Y.u32,null),z.mat2x3u=new z("mat2x3",Y.u32,null),z.mat2x4u=new z("mat2x4",Y.u32,null),z.mat3x2u=new z("mat3x2",Y.u32,null),z.mat3x3u=new z("mat3x3",Y.u32,null),z.mat3x4u=new z("mat3x4",Y.u32,null),z.mat4x2u=new z("mat4x2",Y.u32,null),z.mat4x3u=new z("mat4x3",Y.u32,null),z.mat4x4u=new z("mat4x4",Y.u32,null);class Ta extends Y{constructor(e,t,n,r){super(e),this.storage=t,this.type=n,this.access=r}get astNodeType(){return"pointer"}}class ao extends Y{constructor(e,t,n,r){super(e),this.attributes=t,this.format=n,this.count=r}get astNodeType(){return"array"}get isArray(){return!0}}class qi extends Y{constructor(e,t,n){super(e),this.format=t,this.access=n}get astNodeType(){return"sampler"}}class Dn extends dn{constructor(){super(),this.postfix=null}}class Cr extends Dn{constructor(e){super(),this.value=e}get astNodeType(){return"stringExpr"}toString(){return this.value}constEvaluateString(){return this.value}}class Hn extends Dn{constructor(e,t){super(),this.type=e,this.args=t}get astNodeType(){return"createExpr"}search(e){if(e(this),this.args)for(const t of this.args)t.search(e)}constEvaluate(e,t){return t&&(t[0]=this.type),e.evalExpression(this,e.context)}}class vh extends Dn{constructor(e,t){super(),this.cachedReturnValue=null,this.name=e,this.args=t}get astNodeType(){return"callExpr"}setCachedReturnValue(e){this.cachedReturnValue=e}get isBuiltin(){return wy.has(this.name)}constEvaluate(e,t){return e.evalExpression(this,e.context)}search(e){for(const t of this.args)t.search(e);e(this)}}class Yt extends Dn{constructor(e){super(),this.name=e}get astNodeType(){return"varExpr"}search(e){e(this),this.postfix&&this.postfix.search(e)}constEvaluate(e,t){return e.evalExpression(this,e.context)}}class $y extends Dn{constructor(e,t){super(),this.name=e,this.initializer=t}get astNodeType(){return"constExpr"}constEvaluate(e,t){if(this.initializer){const n=e.evalExpression(this.initializer,e.context);return n!==null&&this.postfix?n.getSubData(e,this.postfix,e.context):n}return null}search(e){this.initializer.search(e)}}class et extends Dn{constructor(e,t){super(),this.value=e,this.type=t}get astNodeType(){return"literalExpr"}constEvaluate(e,t){return t!==void 0&&(t[0]=this.type),this.value}get isScalar(){return this.value instanceof B}get isVector(){return this.value instanceof P||this.value instanceof de}get scalarValue(){return this.value instanceof B?this.value.value:(console.error("Value is not scalar."),0)}get vectorValue(){return this.value instanceof P||this.value instanceof de?this.value.data:(console.error("Value is not a vector or matrix."),new Float32Array(0))}}class Dy extends Dn{constructor(e,t){super(),this.type=e,this.value=t}get astNodeType(){return"bitcastExpr"}search(e){this.value.search(e)}}class gi extends Dn{constructor(e){super(),this.index=e}search(e){this.index.search(e)}}class Oy extends Dn{constructor(){super()}}class Xe extends Oy{constructor(e,t){super(),this.operator=e,this.right=t}get astNodeType(){return"unaryOp"}constEvaluate(e,t){return e.evalExpression(this,e.context)}search(e){this.right.search(e)}}class bn extends Oy{constructor(e,t,n){super(),this.operator=e,this.left=t,this.right=n}get astNodeType(){return"binaryOp"}_getPromotedType(e,t){return e.name===t.name?e:e.name==="f32"||t.name==="f32"?Y.f32:e.name==="u32"||t.name==="u32"?Y.u32:Y.i32}constEvaluate(e,t){return e.evalExpression(this,e.context)}search(e){this.left.search(e),this.right.search(e)}}class My extends dn{constructor(e){super(),this.body=e}search(e){e(this),this.searchBlock(this.body,e)}}class Aa extends Dn{constructor(){super()}get astNodeType(){return"default"}}class Py extends My{constructor(e,t){super(t),this.selectors=e}get astNodeType(){return"case"}search(e){this.searchBlock(this.body,e)}}class Ry extends My{constructor(e){super(e)}get astNodeType(){return"default"}search(e){this.searchBlock(this.body,e)}}class kd extends dn{constructor(e,t,n){super(),this.name=e,this.type=t,this.attributes=n}get astNodeType(){return"argument"}}class PA extends dn{constructor(e,t){super(),this.condition=e,this.body=t}get astNodeType(){return"elseif"}search(e){this.condition.search(e),this.searchBlock(this.body,e)}}class Id extends dn{constructor(e,t,n){super(),this.name=e,this.type=t,this.attributes=n}get astNodeType(){return"member"}}class Ly extends dn{constructor(e,t){super(),this.name=e,this.value=t}get astNodeType(){return"attribute"}}class un{constructor(e,t){this.parent=null,this.typeInfo=e,this.parent=t,this.id=un._id++}clone(){throw"Clone: Not implemented for "+this.constructor.name}setDataValue(e,t,n,r){console.error("SetDataValue: Not implemented for "+this.constructor.name)}getSubData(e,t,n){return console.error("GetDataValue: Not implemented for "+this.constructor.name),null}toString(){return`<${this.typeInfo.getTypeName()}>`}}un._id=0;class sc extends un{constructor(){super(new an("void",null),null)}toString(){return"void"}}sc.void=new sc;class Mr extends un{constructor(e){super(new tc("pointer",e.typeInfo,null),null),this.reference=e}clone(){return this}setDataValue(e,t,n,r){this.reference.setDataValue(e,t,n,r)}getSubData(e,t,n){return t?this.reference.getSubData(e,t,n):this}toString(){return"&"+this.reference.toString()}}class B extends un{constructor(e,t,n=null){super(t,n),e instanceof Int32Array||e instanceof Uint32Array||e instanceof Float32Array?this.data=e:this.typeInfo.name==="x32"?this.data=e-Math.floor(e)!=0?new Float32Array([e]):e>=0?new Uint32Array([e]):new Int32Array([e]):this.typeInfo.name==="i32"||this.typeInfo.name==="bool"?this.data=new Int32Array([e]):this.typeInfo.name==="u32"?this.data=new Uint32Array([e]):this.typeInfo.name==="f32"||this.typeInfo.name==="f16"?this.data=new Float32Array([e]):console.error("ScalarData2: Invalid type",t)}clone(){if(this.data instanceof Float32Array)return new B(new Float32Array(this.data),this.typeInfo,null);if(this.data instanceof Int32Array)return new B(new Int32Array(this.data),this.typeInfo,null);if(this.data instanceof Uint32Array)return new B(new Uint32Array(this.data),this.typeInfo,null);throw"ScalarData: Invalid data type"}get value(){return this.data[0]}set value(e){this.data[0]=e}setDataValue(e,t,n,r){if(n)return void console.error("SetDataValue: Scalar data does not support postfix",n);if(!(t instanceof B))return void console.error("SetDataValue: Invalid value",t);let i=t.data[0];this.typeInfo.name==="i32"||this.typeInfo.name==="u32"?i=Math.floor(i):this.typeInfo.name==="bool"&&(i=i?1:0),this.data[0]=i}getSubData(e,t,n){return t?(console.error("getSubData: Scalar data does not support postfix",t),null):this}toString(){return""+this.value}}function RA(s,e,t){const n=e.length;return n===2?t==="f32"?new P(new Float32Array(e),s.getTypeInfo("vec2f")):t==="i32"||t==="bool"?new P(new Int32Array(e),s.getTypeInfo("vec2i")):t==="u32"?new P(new Uint32Array(e),s.getTypeInfo("vec2u")):t==="f16"?new P(new Float32Array(e),s.getTypeInfo("vec2h")):(console.error("getSubData: Unknown format "+t),null):n===3?t==="f32"?new P(new Float32Array(e),s.getTypeInfo("vec3f")):t==="i32"||t==="bool"?new P(new Int32Array(e),s.getTypeInfo("vec3i")):t==="u32"?new P(new Uint32Array(e),s.getTypeInfo("vec3u")):t==="f16"?new P(new Float32Array(e),s.getTypeInfo("vec3h")):(console.error("getSubData: Unknown format "+t),null):n===4?t==="f32"?new P(new Float32Array(e),s.getTypeInfo("vec4f")):t==="i32"||t==="bool"?new P(new Int32Array(e),s.getTypeInfo("vec4i")):t==="u32"?new P(new Uint32Array(e),s.getTypeInfo("vec4u")):t==="f16"?new P(new Float32Array(e),s.getTypeInfo("vec4h")):(console.error("getSubData: Unknown format "+t),null):(console.error("getSubData: Invalid vector size "+e.length),null)}class P extends un{constructor(e,t,n=null){if(super(t,n),e instanceof Float32Array||e instanceof Uint32Array||e instanceof Int32Array)this.data=e;else{const r=this.typeInfo.name;r==="vec2f"||r==="vec3f"||r==="vec4f"?this.data=new Float32Array(e):r==="vec2i"||r==="vec3i"||r==="vec4i"?this.data=new Int32Array(e):r==="vec2u"||r==="vec3u"||r==="vec4u"?this.data=new Uint32Array(e):r==="vec2h"||r==="vec3h"||r==="vec4h"?this.data=new Float32Array(e):r==="vec2b"||r==="vec3b"||r==="vec4b"?this.data=new Int32Array(e):r==="vec2"||r==="vec3"||r==="vec4"?this.data=new Float32Array(e):console.error("VectorData: Invalid type "+r)}}clone(){if(this.data instanceof Float32Array)return new P(new Float32Array(this.data),this.typeInfo,null);if(this.data instanceof Int32Array)return new P(new Int32Array(this.data),this.typeInfo,null);if(this.data instanceof Uint32Array)return new P(new Uint32Array(this.data),this.typeInfo,null);throw"VectorData: Invalid data type"}setDataValue(e,t,n,r){n instanceof Cr?console.error("TODO: Set vector postfix"):t instanceof P?this.data=t.data:console.error("SetDataValue: Invalid value",t)}getSubData(e,t,n){if(t===null)return this;let r=e.getTypeInfo("f32");if(this.typeInfo instanceof Ar)r=this.typeInfo.format||r;else{const o=this.typeInfo.name;o==="vec2f"||o==="vec3f"||o==="vec4f"?r=e.getTypeInfo("f32"):o==="vec2i"||o==="vec3i"||o==="vec4i"?r=e.getTypeInfo("i32"):o==="vec2b"||o==="vec3b"||o==="vec4b"?r=e.getTypeInfo("bool"):o==="vec2u"||o==="vec3u"||o==="vec4u"?r=e.getTypeInfo("u32"):o==="vec2h"||o==="vec3h"||o==="vec4h"?r=e.getTypeInfo("f16"):console.error("GetSubData: Unknown type "+o)}let i=this;for(;t!==null&&i!==null;){if(t instanceof gi){const o=t.index;let a=-1;if(o instanceof et){if(!(o.value instanceof B))return console.error("GetSubData: Invalid array index "+o.value),null;a=o.value.value}else{const l=e.evalExpression(o,n);if(!(l instanceof B))return console.error("GetSubData: Unknown index type",o),null;a=l.value}if(a<0||a>=i.data.length)return console.error("GetSubData: Index out of range",a),null;if(i.data instanceof Float32Array){const l=new Float32Array(i.data.buffer,i.data.byteOffset+4*a,1);return new B(l,r)}if(i.data instanceof Int32Array){const l=new Int32Array(i.data.buffer,i.data.byteOffset+4*a,1);return new B(l,r)}if(i.data instanceof Uint32Array){const l=new Uint32Array(i.data.buffer,i.data.byteOffset+4*a,1);return new B(l,r)}throw"GetSubData: Invalid data type"}if(!(t instanceof Cr))return console.error("GetSubData: Unknown postfix",t),null;{const o=t.value.toLowerCase();if(o.length===1){let l=0;if(o==="x"||o==="r")l=0;else if(o==="y"||o==="g")l=1;else if(o==="z"||o==="b")l=2;else{if(o!=="w"&&o!=="a")return console.error("GetSubData: Unknown member "+o),null;l=3}if(this.data instanceof Float32Array){let u=new Float32Array(this.data.buffer,this.data.byteOffset+4*l,1);return new B(u,r,this)}if(this.data instanceof Int32Array){let u=new Int32Array(this.data.buffer,this.data.byteOffset+4*l,1);return new B(u,r,this)}if(this.data instanceof Uint32Array){let u=new Uint32Array(this.data.buffer,this.data.byteOffset+4*l,1);return new B(u,r,this)}}const a=[];for(const l of o)l==="x"||l==="r"?a.push(this.data[0]):l==="y"||l==="g"?a.push(this.data[1]):l==="z"||l==="b"?a.push(this.data[2]):l==="w"||l==="a"?a.push(this.data[3]):console.error("GetDataValue: Unknown member "+l);i=RA(e,a,r.name)}t=t.postfix}return i}toString(){let e=""+this.data[0];for(let t=1;t<this.data.length;++t)e+=", "+this.data[t];return e}}class de extends un{constructor(e,t,n=null){super(t,n),e instanceof Float32Array?this.data=e:this.data=new Float32Array(e)}clone(){return new de(new Float32Array(this.data),this.typeInfo,null)}setDataValue(e,t,n,r){n instanceof Cr?console.error("TODO: Set matrix postfix"):t instanceof de?this.data=t.data:console.error("SetDataValue: Invalid value",t)}getSubData(e,t,n){if(t===null)return this;const r=this.typeInfo.name;if(e.getTypeInfo("f32"),this.typeInfo instanceof Ar)this.typeInfo.format;else if(r.endsWith("f"))e.getTypeInfo("f32");else if(r.endsWith("i"))e.getTypeInfo("i32");else if(r.endsWith("u"))e.getTypeInfo("u32");else{if(!r.endsWith("h"))return console.error("GetDataValue: Unknown type "+r),null;e.getTypeInfo("f16")}if(t instanceof gi){const i=t.index;let o=-1;if(i instanceof et){if(!(i.value instanceof B))return console.error("GetDataValue: Invalid array index "+i.value),null;o=i.value.value}else{const u=e.evalExpression(i,n);if(!(u instanceof B))return console.error("GetDataValue: Unknown index type",i),null;o=u.value}if(o<0||o>=this.data.length)return console.error("GetDataValue: Index out of range",o),null;const a=r.endsWith("h")?"h":"f";let l;if(r==="mat2x2"||r==="mat2x2f"||r==="mat2x2h"||r==="mat3x2"||r==="mat3x2f"||r==="mat3x2h"||r==="mat4x2"||r==="mat4x2f"||r==="mat4x2h")l=new P(new Float32Array(this.data.buffer,this.data.byteOffset+8*o,2),e.getTypeInfo("vec2"+a));else if(r==="mat2x3"||r==="mat2x3f"||r==="mat2x3h"||r==="mat3x3"||r==="mat3x3f"||r==="mat3x3h"||r==="mat4x3"||r==="mat4x3f"||r==="mat4x3h")l=new P(new Float32Array(this.data.buffer,this.data.byteOffset+12*o,3),e.getTypeInfo("vec3"+a));else{if(r!=="mat2x4"&&r!=="mat2x4f"&&r!=="mat2x4h"&&r!=="mat3x4"&&r!=="mat3x4f"&&r!=="mat3x4h"&&r!=="mat4x4"&&r!=="mat4x4f"&&r!=="mat4x4h")return console.error("GetDataValue: Unknown type "+r),null;l=new P(new Float32Array(this.data.buffer,this.data.byteOffset+16*o,4),e.getTypeInfo("vec4"+a))}return t.postfix?l.getSubData(e,t.postfix,n):l}return console.error("GetDataValue: Invalid postfix",t),null}toString(){let e=""+this.data[0];for(let t=1;t<this.data.length;++t)e+=", "+this.data[t];return e}}class qe extends un{constructor(e,t,n=0,r=null){super(t,r),this.buffer=e instanceof ArrayBuffer?e:e.buffer,this.offset=n}clone(){const e=new Uint8Array(new Uint8Array(this.buffer,this.offset,this.typeInfo.size));return new qe(e.buffer,this.typeInfo,0,null)}setDataValue(e,t,n,r){if(t===null)return void console.log("setDataValue: NULL data.");let i=this.offset,o=this.typeInfo;for(;n;){if(n instanceof gi)if(o instanceof Ps){const a=n.index;if(a instanceof et){if(!(a.value instanceof B))return void console.error("SetDataValue: Invalid index type "+a.value);i+=a.value.value*o.stride}else{const l=e.evalExpression(a,r);if(!(l instanceof B))return void console.error("SetDataValue: Unknown index type",a);i+=l.value*o.stride}o=o.format}else console.error(`SetDataValue: Type ${o.getTypeName()} is not an array`);else{if(!(n instanceof Cr))return void console.error("SetDataValue: Unknown postfix type",n);{const a=n.value;if(o instanceof Ns){let l=!1;for(const u of o.members)if(u.name===a){i+=u.offset,o=u.type,l=!0;break}if(!l)return void console.error(`SetDataValue: Member ${a} not found`)}else if(o instanceof an){const l=o.getTypeName();let u=0;if(a==="x"||a==="r")u=0;else if(a==="y"||a==="g")u=1;else if(a==="z"||a==="b")u=2;else{if(a!=="w"&&a!=="a")return void console.error("SetDataValue: Unknown member "+a);u=3}if(!(t instanceof B))return void console.error("SetDataValue: Invalid value",t);const c=t.value;return l==="vec2f"?void(new Float32Array(this.buffer,i,2)[u]=c):l==="vec3f"?void(new Float32Array(this.buffer,i,3)[u]=c):l==="vec4f"?void(new Float32Array(this.buffer,i,4)[u]=c):l==="vec2i"?void(new Int32Array(this.buffer,i,2)[u]=c):l==="vec3i"?void(new Int32Array(this.buffer,i,3)[u]=c):l==="vec4i"?void(new Int32Array(this.buffer,i,4)[u]=c):l==="vec2u"?void(new Uint32Array(this.buffer,i,2)[u]=c):l==="vec3u"?void(new Uint32Array(this.buffer,i,3)[u]=c):l==="vec4u"?void(new Uint32Array(this.buffer,i,4)[u]=c):void console.error(`SetDataValue: Type ${l} is not a struct`)}}}n=n.postfix}this.setData(e,t,o,i,r)}setData(e,t,n,r,i){const o=n.getTypeName();if(o!=="f32"&&o!=="f16")if(o!=="i32"&&o!=="atomic<i32>"&&o!=="x32")if(o!=="u32"&&o!=="atomic<u32>")if(o!=="bool")if(o!=="vec2f"&&o!=="vec2h")if(o!=="vec3f"&&o!=="vec3h")if(o!=="vec4f"&&o!=="vec4h")if(o!=="vec2i")if(o!=="vec3i")if(o!=="vec4i")if(o!=="vec2u")if(o!=="vec3u")if(o!=="vec4u")if(o!=="vec2b")if(o!=="vec3b")if(o!=="vec4b")if(o!=="mat2x2f"&&o!=="mat2x2h")if(o!=="mat2x3f"&&o!=="mat2x3h")if(o!=="mat2x4f"&&o!=="mat2x4h")if(o!=="mat3x2f"&&o!=="mat3x2h")if(o!=="mat3x3f"&&o!=="mat3x3h")if(o!=="mat3x4f"&&o!=="mat3x4h")if(o!=="mat4x2f"&&o!=="mat4x2h")if(o!=="mat4x3f"&&o!=="mat4x3h")if(o!=="mat4x4f"&&o!=="mat4x4h")if(t instanceof qe){if(n===t.typeInfo)return void new Uint8Array(this.buffer,r,t.buffer.byteLength).set(new Uint8Array(t.buffer));console.error("SetDataValue: Type mismatch",o,t.typeInfo.getTypeName())}else console.error("SetData: Unknown type "+o);else{const a=new Float32Array(this.buffer,r,16);t instanceof de?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3],a[4]=t.data[4],a[5]=t.data[5],a[6]=t.data[6],a[7]=t.data[7],a[8]=t.data[8],a[9]=t.data[9],a[10]=t.data[10],a[11]=t.data[11],a[12]=t.data[12],a[13]=t.data[13],a[14]=t.data[14],a[15]=t.data[15]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3],a[4]=t[4],a[5]=t[5],a[6]=t[6],a[7]=t[7],a[8]=t[8],a[9]=t[9],a[10]=t[10],a[11]=t[11],a[12]=t[12],a[13]=t[13],a[14]=t[14],a[15]=t[15])}else{const a=new Float32Array(this.buffer,r,12);t instanceof de?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3],a[4]=t.data[4],a[5]=t.data[5],a[6]=t.data[6],a[7]=t.data[7],a[8]=t.data[8],a[9]=t.data[9],a[10]=t.data[10],a[11]=t.data[11]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3],a[4]=t[4],a[5]=t[5],a[6]=t[6],a[7]=t[7],a[8]=t[8],a[9]=t[9],a[10]=t[10],a[11]=t[11])}else{const a=new Float32Array(this.buffer,r,8);t instanceof de?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3],a[4]=t.data[4],a[5]=t.data[5],a[6]=t.data[6],a[7]=t.data[7]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3],a[4]=t[4],a[5]=t[5],a[6]=t[6],a[7]=t[7])}else{const a=new Float32Array(this.buffer,r,12);t instanceof de?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3],a[4]=t.data[4],a[5]=t.data[5],a[6]=t.data[6],a[7]=t.data[7],a[8]=t.data[8],a[9]=t.data[9],a[10]=t.data[10],a[11]=t.data[11]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3],a[4]=t[4],a[5]=t[5],a[6]=t[6],a[7]=t[7],a[8]=t[8],a[9]=t[9],a[10]=t[10],a[11]=t[11])}else{const a=new Float32Array(this.buffer,r,9);t instanceof de?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3],a[4]=t.data[4],a[5]=t.data[5],a[6]=t.data[6],a[7]=t.data[7],a[8]=t.data[8]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3],a[4]=t[4],a[5]=t[5],a[6]=t[6],a[7]=t[7],a[8]=t[8])}else{const a=new Float32Array(this.buffer,r,6);t instanceof de?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3],a[4]=t.data[4],a[5]=t.data[5]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3],a[4]=t[4],a[5]=t[5])}else{const a=new Float32Array(this.buffer,r,8);t instanceof de?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3],a[4]=t.data[4],a[5]=t.data[5],a[6]=t.data[6],a[7]=t.data[7]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3],a[4]=t[4],a[5]=t[5],a[6]=t[6],a[7]=t[7])}else{const a=new Float32Array(this.buffer,r,6);t instanceof de?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3],a[4]=t.data[4],a[5]=t.data[5]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3],a[4]=t[4],a[5]=t[5])}else{const a=new Float32Array(this.buffer,r,4);t instanceof de?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3])}else{const a=new Uint32Array(this.buffer,r,4);t instanceof P?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3])}else{const a=new Uint32Array(this.buffer,r,3);t instanceof P?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2]):(a[0]=t[0],a[1]=t[1],a[2]=t[2])}else{const a=new Uint32Array(this.buffer,r,2);t instanceof P?(a[0]=t.data[0],a[1]=t.data[1]):(a[0]=t[0],a[1]=t[1])}else{const a=new Uint32Array(this.buffer,r,4);t instanceof P?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3])}else{const a=new Uint32Array(this.buffer,r,3);t instanceof P?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2]):(a[0]=t[0],a[1]=t[1],a[2]=t[2])}else{const a=new Uint32Array(this.buffer,r,2);t instanceof P?(a[0]=t.data[0],a[1]=t.data[1]):(a[0]=t[0],a[1]=t[1])}else{const a=new Int32Array(this.buffer,r,4);t instanceof P?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3])}else{const a=new Int32Array(this.buffer,r,3);t instanceof P?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2]):(a[0]=t[0],a[1]=t[1],a[2]=t[2])}else{const a=new Int32Array(this.buffer,r,2);t instanceof P?(a[0]=t.data[0],a[1]=t.data[1]):(a[0]=t[0],a[1]=t[1])}else{const a=new Float32Array(this.buffer,r,4);t instanceof P?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3])}else{const a=new Float32Array(this.buffer,r,3);t instanceof P?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2]):(a[0]=t[0],a[1]=t[1],a[2]=t[2])}else{const a=new Float32Array(this.buffer,r,2);t instanceof P?(a[0]=t.data[0],a[1]=t.data[1]):(a[0]=t[0],a[1]=t[1])}else t instanceof B&&(new Int32Array(this.buffer,r,1)[0]=t.value);else t instanceof B&&(new Uint32Array(this.buffer,r,1)[0]=t.value);else t instanceof B&&(new Int32Array(this.buffer,r,1)[0]=t.value);else t instanceof B&&(new Float32Array(this.buffer,r,1)[0]=t.value)}getSubData(e,t,n){var r,i,o;if(t===null)return this;let a=this.offset,l=this.typeInfo;for(;t;){if(t instanceof gi){const c=t.index,h=c instanceof Dn?e.evalExpression(c,n):c;let d=0;if(h instanceof B?d=h.value:typeof h=="number"?d=h:console.error("GetDataValue: Invalid index type",c),l instanceof Ps)a+=d*l.stride,l=l.format;else{const w=l.getTypeName();w==="mat4x4"||w==="mat4x4f"||w==="mat4x4h"?(a+=16*d,l=e.getTypeInfo("vec4f")):console.error(`getDataValue: Type ${l.getTypeName()} is not an array`)}}else{if(!(t instanceof Cr))return console.error("GetDataValue: Unknown postfix type",t),null;{const c=t.value;if(l instanceof Ns){let h=!1;for(const d of l.members)if(d.name===c){a+=d.offset,l=d.type,h=!0;break}if(!h)return console.error(`GetDataValue: Member ${c} not found`),null}else if(l instanceof an){const h=l.getTypeName();if(h==="vec2f"||h==="vec3f"||h==="vec4f"||h==="vec2i"||h==="vec3i"||h==="vec4i"||h==="vec2u"||h==="vec3u"||h==="vec4u"||h==="vec2b"||h==="vec3b"||h==="vec4b"||h==="vec2h"||h==="vec3h"||h==="vec4h"||h==="vec2"||h==="vec3"||h==="vec4"){if(c.length>0&&c.length<5){let d="f";const w=[];for(let k=0;k<c.length;++k){const A=c[k].toLowerCase();let m=0;if(A==="x"||A==="r")m=0;else if(A==="y"||A==="g")m=1;else if(A==="z"||A==="b")m=2;else{if(A!=="w"&&A!=="a")return console.error("Unknown member "+c),null;m=3}if(c.length===1){if(h.endsWith("f"))return this.buffer.byteLength<a+4*m+4?(console.log("Insufficient buffer data"),null):new B(new Float32Array(this.buffer,a+4*m,1),e.getTypeInfo("f32"),this);if(h.endsWith("h"))return new B(new Float32Array(this.buffer,a+4*m,1),e.getTypeInfo("f16"),this);if(h.endsWith("i"))return new B(new Int32Array(this.buffer,a+4*m,1),e.getTypeInfo("i32"),this);if(h.endsWith("b"))return new B(new Int32Array(this.buffer,a+4*m,1),e.getTypeInfo("bool"),this);if(h.endsWith("u"))return new B(new Uint32Array(this.buffer,a+4*m,1),e.getTypeInfo("i32"),this)}if(h==="vec2f")w.push(new Float32Array(this.buffer,a,2)[m]);else if(h==="vec3f"){if(a+12>=this.buffer.byteLength)return console.log("Insufficient buffer data"),null;const S=new Float32Array(this.buffer,a,3);w.push(S[m])}else if(h==="vec4f")w.push(new Float32Array(this.buffer,a,4)[m]);else if(h==="vec2i")d="i",w.push(new Int32Array(this.buffer,a,2)[m]);else if(h==="vec3i")d="i",w.push(new Int32Array(this.buffer,a,3)[m]);else if(h==="vec4i")d="i",w.push(new Int32Array(this.buffer,a,4)[m]);else if(h==="vec2u"){d="u";const S=new Uint32Array(this.buffer,a,2);w.push(S[m])}else h==="vec3u"?(d="u",w.push(new Uint32Array(this.buffer,a,3)[m])):h==="vec4u"&&(d="u",w.push(new Uint32Array(this.buffer,a,4)[m]))}return w.length===2?l=e.getTypeInfo("vec2"+d):w.length===3?l=e.getTypeInfo("vec3"+d):w.length===4?l=e.getTypeInfo("vec4"+d):console.error("GetDataValue: Invalid vector length "+w.length),new P(w,l,null)}return console.error("GetDataValue: Unknown member "+c),null}return console.error(`GetDataValue: Type ${h} is not a struct`),null}}}t=t.postfix}const u=l.getTypeName();return u==="f32"?new B(new Float32Array(this.buffer,a,1),l,this):u==="i32"?new B(new Int32Array(this.buffer,a,1),l,this):u==="u32"?new B(new Uint32Array(this.buffer,a,1),l,this):u==="vec2f"?new P(new Float32Array(this.buffer,a,2),l,this):u==="vec3f"?new P(new Float32Array(this.buffer,a,3),l,this):u==="vec4f"?new P(new Float32Array(this.buffer,a,4),l,this):u==="vec2i"?new P(new Int32Array(this.buffer,a,2),l,this):u==="vec3i"?new P(new Int32Array(this.buffer,a,3),l,this):u==="vec4i"?new P(new Int32Array(this.buffer,a,4),l,this):u==="vec2u"?new P(new Uint32Array(this.buffer,a,2),l,this):u==="vec3u"?new P(new Uint32Array(this.buffer,a,3),l,this):u==="vec4u"?new P(new Uint32Array(this.buffer,a,4),l,this):l instanceof Ar&&l.name==="atomic"?((r=l.format)===null||r===void 0?void 0:r.name)==="u32"?new B(new Uint32Array(this.buffer,a,1)[0],l.format,this):((i=l.format)===null||i===void 0?void 0:i.name)==="i32"?new B(new Int32Array(this.buffer,a,1)[0],l.format,this):(console.error("GetDataValue: Invalid atomic format "+((o=l.format)===null||o===void 0?void 0:o.name)),null):new qe(this.buffer,l,a,this)}toString(){let e="";if(this.typeInfo instanceof Ps)if(this.typeInfo.format.name==="f32"){const t=new Float32Array(this.buffer,this.offset);e="["+t[0];for(let n=1;n<t.length;++n)e+=", "+t[n]}else if(this.typeInfo.format.name==="i32"){const t=new Int32Array(this.buffer,this.offset);e="["+t[0];for(let n=1;n<t.length;++n)e+=", "+t[n]}else if(this.typeInfo.format.name==="u32"){const t=new Uint32Array(this.buffer,this.offset);e="["+t[0];for(let n=1;n<t.length;++n)e+=", "+t[n]}else if(this.typeInfo.format.name==="vec2f"){const t=new Float32Array(this.buffer,this.offset);e=`[${t[0]}, ${t[1]}]`;for(let n=1;n<t.length/2;++n)e+=`, [${t[2*n]}, ${t[2*n+1]}]`}else if(this.typeInfo.format.name==="vec3f"){const t=new Float32Array(this.buffer,this.offset);e=`[${t[0]}, ${t[1]}, ${t[2]}]`;for(let n=4;n<t.length;n+=4)e+=`, [${t[n]}, ${t[n+1]}, ${t[n+2]}]`}else if(this.typeInfo.format.name==="vec4f"){const t=new Float32Array(this.buffer,this.offset);e=`[${t[0]}, ${t[1]}, ${t[2]}, ${t[3]}]`;for(let n=4;n<t.length;n+=4)e+=`, [${t[n]}, ${t[n+1]}, ${t[n+2]}, ${t[n+3]}]`}else e="[...]";else this.typeInfo instanceof Ns?e+="{...}":e="[...]";return e}}class hs extends un{constructor(e,t,n,r){super(t,null),this.data=e,this.descriptor=n,this.view=r}clone(){return new hs(this.data,this.typeInfo,this.descriptor,this.view)}get width(){var e,t;const n=this.descriptor.size;return n instanceof Array&&n.length>0?(e=n[0])!==null&&e!==void 0?e:0:n instanceof Object&&(t=n.width)!==null&&t!==void 0?t:0}get height(){var e,t;const n=this.descriptor.size;return n instanceof Array&&n.length>1?(e=n[1])!==null&&e!==void 0?e:0:n instanceof Object&&(t=n.height)!==null&&t!==void 0?t:0}get depthOrArrayLayers(){var e,t;const n=this.descriptor.size;return n instanceof Array&&n.length>2?(e=n[2])!==null&&e!==void 0?e:0:n instanceof Object&&(t=n.depthOrArrayLayers)!==null&&t!==void 0?t:0}get format(){var e;return this.descriptor&&(e=this.descriptor.format)!==null&&e!==void 0?e:"rgba8unorm"}get sampleCount(){var e;return this.descriptor&&(e=this.descriptor.sampleCount)!==null&&e!==void 0?e:1}get mipLevelCount(){var e;return this.descriptor&&(e=this.descriptor.mipLevelCount)!==null&&e!==void 0?e:1}get dimension(){var e;return this.descriptor&&(e=this.descriptor.dimension)!==null&&e!==void 0?e:"2d"}getMipLevelSize(e){if(e>=this.mipLevelCount)return[0,0,0];const t=[this.width,this.height,this.depthOrArrayLayers];for(let n=0;n<t.length;++n)t[n]=Math.max(1,t[n]>>e);return t}get texelByteSize(){const e=this.format,t=yu[e];return t?t.isDepthStencil?4:t.bytesPerBlock:0}get bytesPerRow(){return this.width*this.texelByteSize}get isDepthStencil(){const e=this.format,t=yu[e];return!!t&&t.isDepthStencil}getGpuSize(){const e=this.format,t=yu[e],n=this.width;if(!e||n<=0||!t)return-1;const r=this.height,i=this.depthOrArrayLayers,o=this.dimension;return n/t.blockWidth*(o==="1d"?1:r/t.blockHeight)*t.bytesPerBlock*i}getPixel(e,t,n=0,r=0){const i=this.texelByteSize,o=this.bytesPerRow,a=this.height,l=this.data[r];return((u,c,h,d,w,k,A,m)=>{const S=d*(A>>=w)*(k>>=w)+h*A+c*m;switch(this.format){case"r8unorm":return[ve(u,S,"8unorm",1)[0]];case"r8snorm":return[ve(u,S,"8snorm",1)[0]];case"r8uint":return[ve(u,S,"8uint",1)[0]];case"r8sint":return[ve(u,S,"8sint",1)[0]];case"rg8unorm":{const b=ve(u,S,"8unorm",2);return[b[0],b[1]]}case"rg8snorm":{const b=ve(u,S,"8snorm",2);return[b[0],b[1]]}case"rg8uint":{const b=ve(u,S,"8uint",2);return[b[0],b[1]]}case"rg8sint":{const b=ve(u,S,"8sint",2);return[b[0],b[1]]}case"rgba8unorm-srgb":case"rgba8unorm":{const b=ve(u,S,"8unorm",4);return[b[0],b[1],b[2],b[3]]}case"rgba8snorm":{const b=ve(u,S,"8snorm",4);return[b[0],b[1],b[2],b[3]]}case"rgba8uint":{const b=ve(u,S,"8uint",4);return[b[0],b[1],b[2],b[3]]}case"rgba8sint":{const b=ve(u,S,"8sint",4);return[b[0],b[1],b[2],b[3]]}case"bgra8unorm-srgb":case"bgra8unorm":{const b=ve(u,S,"8unorm",4);return[b[2],b[1],b[0],b[3]]}case"r16uint":return[ve(u,S,"16uint",1)[0]];case"r16sint":return[ve(u,S,"16sint",1)[0]];case"r16float":return[ve(u,S,"16float",1)[0]];case"rg16uint":{const b=ve(u,S,"16uint",2);return[b[0],b[1]]}case"rg16sint":{const b=ve(u,S,"16sint",2);return[b[0],b[1]]}case"rg16float":{const b=ve(u,S,"16float",2);return[b[0],b[1]]}case"rgba16uint":{const b=ve(u,S,"16uint",4);return[b[0],b[1],b[2],b[3]]}case"rgba16sint":{const b=ve(u,S,"16sint",4);return[b[0],b[1],b[2],b[3]]}case"rgba16float":{const b=ve(u,S,"16float",4);return[b[0],b[1],b[2],b[3]]}case"r32uint":return[ve(u,S,"32uint",1)[0]];case"r32sint":return[ve(u,S,"32sint",1)[0]];case"depth16unorm":case"depth24plus":case"depth24plus-stencil8":case"depth32float":case"depth32float-stencil8":case"r32float":return[ve(u,S,"32float",1)[0]];case"rg32uint":{const b=ve(u,S,"32uint",2);return[b[0],b[1]]}case"rg32sint":{const b=ve(u,S,"32sint",2);return[b[0],b[1]]}case"rg32float":{const b=ve(u,S,"32float",2);return[b[0],b[1]]}case"rgba32uint":{const b=ve(u,S,"32uint",4);return[b[0],b[1],b[2],b[3]]}case"rgba32sint":{const b=ve(u,S,"32sint",4);return[b[0],b[1],b[2],b[3]]}case"rgba32float":{const b=ve(u,S,"32float",4);return[b[0],b[1],b[2],b[3]]}case"rg11b10ufloat":{const b=new Uint32Array(u.buffer,S,1)[0],f=(4192256&b)>>11,v=(4290772992&b)>>22;return[vd(2047&b),vd(f),(_=>{const E=112+(_>>5&31)<<23|(31&_)<<18;return bh[0]=E,by[0]})(v),1]}}return null})(new Uint8Array(l),e,t,n,r,a,o,i)}setPixel(e,t,n,r,i){const o=this.texelByteSize,a=this.bytesPerRow,l=this.height,u=this.data[r];((c,h,d,w,k,A,m,S,b,f)=>{const v=w*(m>>=k)*(A>>=k)+d*m+h*S;switch(b){case"r8unorm":return void Ee(c,v,"8unorm",1,f);case"r8snorm":return void Ee(c,v,"8snorm",1,f);case"r8uint":return void Ee(c,v,"8uint",1,f);case"r8sint":return void Ee(c,v,"8sint",1,f);case"rg8unorm":return void Ee(c,v,"8unorm",2,f);case"rg8snorm":return void Ee(c,v,"8snorm",2,f);case"rg8uint":return void Ee(c,v,"8uint",2,f);case"rg8sint":return void Ee(c,v,"8sint",2,f);case"rgba8unorm-srgb":case"rgba8unorm":case"bgra8unorm-srgb":case"bgra8unorm":return void Ee(c,v,"8unorm",4,f);case"rgba8snorm":return void Ee(c,v,"8snorm",4,f);case"rgba8uint":return void Ee(c,v,"8uint",4,f);case"rgba8sint":return void Ee(c,v,"8sint",4,f);case"r16uint":return void Ee(c,v,"16uint",1,f);case"r16sint":return void Ee(c,v,"16sint",1,f);case"r16float":return void Ee(c,v,"16float",1,f);case"rg16uint":return void Ee(c,v,"16uint",2,f);case"rg16sint":return void Ee(c,v,"16sint",2,f);case"rg16float":return void Ee(c,v,"16float",2,f);case"rgba16uint":return void Ee(c,v,"16uint",4,f);case"rgba16sint":return void Ee(c,v,"16sint",4,f);case"rgba16float":return void Ee(c,v,"16float",4,f);case"r32uint":return void Ee(c,v,"32uint",1,f);case"r32sint":return void Ee(c,v,"32sint",1,f);case"depth16unorm":case"depth24plus":case"depth24plus-stencil8":case"depth32float":case"depth32float-stencil8":case"r32float":return void Ee(c,v,"32float",1,f);case"rg32uint":return void Ee(c,v,"32uint",2,f);case"rg32sint":return void Ee(c,v,"32sint",2,f);case"rg32float":return void Ee(c,v,"32float",2,f);case"rgba32uint":return void Ee(c,v,"32uint",4,f);case"rgba32sint":return void Ee(c,v,"32sint",4,f);case"rgba32float":return void Ee(c,v,"32float",4,f);case"rg11b10ufloat":console.error("TODO: rg11b10ufloat not supported for writing")}})(new Uint8Array(u),e,t,n,r,l,a,o,this.format,i)}}(s=>{s[s.token=0]="token",s[s.keyword=1]="keyword",s[s.reserved=2]="reserved"})(F||(F={}));class U{constructor(e,t,n){this.name=e,this.type=t,this.rule=n}toString(){return this.name}}class O{}G=O,O.none=new U("",F.reserved,""),O.eof=new U("EOF",F.token,""),O.reserved={asm:new U("asm",F.reserved,"asm"),bf16:new U("bf16",F.reserved,"bf16"),do:new U("do",F.reserved,"do"),enum:new U("enum",F.reserved,"enum"),f16:new U("f16",F.reserved,"f16"),f64:new U("f64",F.reserved,"f64"),handle:new U("handle",F.reserved,"handle"),i8:new U("i8",F.reserved,"i8"),i16:new U("i16",F.reserved,"i16"),i64:new U("i64",F.reserved,"i64"),mat:new U("mat",F.reserved,"mat"),premerge:new U("premerge",F.reserved,"premerge"),regardless:new U("regardless",F.reserved,"regardless"),typedef:new U("typedef",F.reserved,"typedef"),u8:new U("u8",F.reserved,"u8"),u16:new U("u16",F.reserved,"u16"),u64:new U("u64",F.reserved,"u64"),unless:new U("unless",F.reserved,"unless"),using:new U("using",F.reserved,"using"),vec:new U("vec",F.reserved,"vec"),void:new U("void",F.reserved,"void")},O.keywords={array:new U("array",F.keyword,"array"),atomic:new U("atomic",F.keyword,"atomic"),bool:new U("bool",F.keyword,"bool"),f32:new U("f32",F.keyword,"f32"),i32:new U("i32",F.keyword,"i32"),mat2x2:new U("mat2x2",F.keyword,"mat2x2"),mat2x3:new U("mat2x3",F.keyword,"mat2x3"),mat2x4:new U("mat2x4",F.keyword,"mat2x4"),mat3x2:new U("mat3x2",F.keyword,"mat3x2"),mat3x3:new U("mat3x3",F.keyword,"mat3x3"),mat3x4:new U("mat3x4",F.keyword,"mat3x4"),mat4x2:new U("mat4x2",F.keyword,"mat4x2"),mat4x3:new U("mat4x3",F.keyword,"mat4x3"),mat4x4:new U("mat4x4",F.keyword,"mat4x4"),ptr:new U("ptr",F.keyword,"ptr"),sampler:new U("sampler",F.keyword,"sampler"),sampler_comparison:new U("sampler_comparison",F.keyword,"sampler_comparison"),struct:new U("struct",F.keyword,"struct"),texture_1d:new U("texture_1d",F.keyword,"texture_1d"),texture_2d:new U("texture_2d",F.keyword,"texture_2d"),texture_2d_array:new U("texture_2d_array",F.keyword,"texture_2d_array"),texture_3d:new U("texture_3d",F.keyword,"texture_3d"),texture_cube:new U("texture_cube",F.keyword,"texture_cube"),texture_cube_array:new U("texture_cube_array",F.keyword,"texture_cube_array"),texture_multisampled_2d:new U("texture_multisampled_2d",F.keyword,"texture_multisampled_2d"),texture_storage_1d:new U("texture_storage_1d",F.keyword,"texture_storage_1d"),texture_storage_2d:new U("texture_storage_2d",F.keyword,"texture_storage_2d"),texture_storage_2d_array:new U("texture_storage_2d_array",F.keyword,"texture_storage_2d_array"),texture_storage_3d:new U("texture_storage_3d",F.keyword,"texture_storage_3d"),texture_depth_2d:new U("texture_depth_2d",F.keyword,"texture_depth_2d"),texture_depth_2d_array:new U("texture_depth_2d_array",F.keyword,"texture_depth_2d_array"),texture_depth_cube:new U("texture_depth_cube",F.keyword,"texture_depth_cube"),texture_depth_cube_array:new U("texture_depth_cube_array",F.keyword,"texture_depth_cube_array"),texture_depth_multisampled_2d:new U("texture_depth_multisampled_2d",F.keyword,"texture_depth_multisampled_2d"),texture_external:new U("texture_external",F.keyword,"texture_external"),u32:new U("u32",F.keyword,"u32"),vec2:new U("vec2",F.keyword,"vec2"),vec3:new U("vec3",F.keyword,"vec3"),vec4:new U("vec4",F.keyword,"vec4"),bitcast:new U("bitcast",F.keyword,"bitcast"),block:new U("block",F.keyword,"block"),break:new U("break",F.keyword,"break"),case:new U("case",F.keyword,"case"),continue:new U("continue",F.keyword,"continue"),continuing:new U("continuing",F.keyword,"continuing"),default:new U("default",F.keyword,"default"),diagnostic:new U("diagnostic",F.keyword,"diagnostic"),discard:new U("discard",F.keyword,"discard"),else:new U("else",F.keyword,"else"),enable:new U("enable",F.keyword,"enable"),fallthrough:new U("fallthrough",F.keyword,"fallthrough"),false:new U("false",F.keyword,"false"),fn:new U("fn",F.keyword,"fn"),for:new U("for",F.keyword,"for"),function:new U("function",F.keyword,"function"),if:new U("if",F.keyword,"if"),let:new U("let",F.keyword,"let"),const:new U("const",F.keyword,"const"),loop:new U("loop",F.keyword,"loop"),while:new U("while",F.keyword,"while"),private:new U("private",F.keyword,"private"),read:new U("read",F.keyword,"read"),read_write:new U("read_write",F.keyword,"read_write"),return:new U("return",F.keyword,"return"),requires:new U("requires",F.keyword,"requires"),storage:new U("storage",F.keyword,"storage"),switch:new U("switch",F.keyword,"switch"),true:new U("true",F.keyword,"true"),alias:new U("alias",F.keyword,"alias"),type:new U("type",F.keyword,"type"),uniform:new U("uniform",F.keyword,"uniform"),var:new U("var",F.keyword,"var"),override:new U("override",F.keyword,"override"),workgroup:new U("workgroup",F.keyword,"workgroup"),write:new U("write",F.keyword,"write"),r8unorm:new U("r8unorm",F.keyword,"r8unorm"),r8snorm:new U("r8snorm",F.keyword,"r8snorm"),r8uint:new U("r8uint",F.keyword,"r8uint"),r8sint:new U("r8sint",F.keyword,"r8sint"),r16uint:new U("r16uint",F.keyword,"r16uint"),r16sint:new U("r16sint",F.keyword,"r16sint"),r16float:new U("r16float",F.keyword,"r16float"),rg8unorm:new U("rg8unorm",F.keyword,"rg8unorm"),rg8snorm:new U("rg8snorm",F.keyword,"rg8snorm"),rg8uint:new U("rg8uint",F.keyword,"rg8uint"),rg8sint:new U("rg8sint",F.keyword,"rg8sint"),r32uint:new U("r32uint",F.keyword,"r32uint"),r32sint:new U("r32sint",F.keyword,"r32sint"),r32float:new U("r32float",F.keyword,"r32float"),rg16uint:new U("rg16uint",F.keyword,"rg16uint"),rg16sint:new U("rg16sint",F.keyword,"rg16sint"),rg16float:new U("rg16float",F.keyword,"rg16float"),rgba8unorm:new U("rgba8unorm",F.keyword,"rgba8unorm"),rgba8unorm_srgb:new U("rgba8unorm_srgb",F.keyword,"rgba8unorm_srgb"),rgba8snorm:new U("rgba8snorm",F.keyword,"rgba8snorm"),rgba8uint:new U("rgba8uint",F.keyword,"rgba8uint"),rgba8sint:new U("rgba8sint",F.keyword,"rgba8sint"),bgra8unorm:new U("bgra8unorm",F.keyword,"bgra8unorm"),bgra8unorm_srgb:new U("bgra8unorm_srgb",F.keyword,"bgra8unorm_srgb"),rgb10a2unorm:new U("rgb10a2unorm",F.keyword,"rgb10a2unorm"),rg11b10float:new U("rg11b10float",F.keyword,"rg11b10float"),rg32uint:new U("rg32uint",F.keyword,"rg32uint"),rg32sint:new U("rg32sint",F.keyword,"rg32sint"),rg32float:new U("rg32float",F.keyword,"rg32float"),rgba16uint:new U("rgba16uint",F.keyword,"rgba16uint"),rgba16sint:new U("rgba16sint",F.keyword,"rgba16sint"),rgba16float:new U("rgba16float",F.keyword,"rgba16float"),rgba32uint:new U("rgba32uint",F.keyword,"rgba32uint"),rgba32sint:new U("rgba32sint",F.keyword,"rgba32sint"),rgba32float:new U("rgba32float",F.keyword,"rgba32float"),static_assert:new U("static_assert",F.keyword,"static_assert")},O.tokens={decimal_float_literal:new U("decimal_float_literal",F.token,/((-?[0-9]*\.[0-9]+|-?[0-9]+\.[0-9]*)((e|E)(\+|-)?[0-9]+)?[fh]?)|(-?[0-9]+(e|E)(\+|-)?[0-9]+[fh]?)|(-?[0-9]+[fh])/),hex_float_literal:new U("hex_float_literal",F.token,/-?0x((([0-9a-fA-F]*\.[0-9a-fA-F]+|[0-9a-fA-F]+\.[0-9a-fA-F]*)((p|P)(\+|-)?[0-9]+[fh]?)?)|([0-9a-fA-F]+(p|P)(\+|-)?[0-9]+[fh]?))/),int_literal:new U("int_literal",F.token,/-?0x[0-9a-fA-F]+|0i?|-?[1-9][0-9]*i?/),uint_literal:new U("uint_literal",F.token,/0x[0-9a-fA-F]+u|0u|[1-9][0-9]*u/),name:new U("name",F.token,/([_\p{XID_Start}][\p{XID_Continue}]+)|([\p{XID_Start}])/u),ident:new U("ident",F.token,/[_a-zA-Z][0-9a-zA-Z_]*/),and:new U("and",F.token,"&"),and_and:new U("and_and",F.token,"&&"),arrow:new U("arrow ",F.token,"->"),attr:new U("attr",F.token,"@"),forward_slash:new U("forward_slash",F.token,"/"),bang:new U("bang",F.token,"!"),bracket_left:new U("bracket_left",F.token,"["),bracket_right:new U("bracket_right",F.token,"]"),brace_left:new U("brace_left",F.token,"{"),brace_right:new U("brace_right",F.token,"}"),colon:new U("colon",F.token,":"),comma:new U("comma",F.token,","),equal:new U("equal",F.token,"="),equal_equal:new U("equal_equal",F.token,"=="),not_equal:new U("not_equal",F.token,"!="),greater_than:new U("greater_than",F.token,">"),greater_than_equal:new U("greater_than_equal",F.token,">="),shift_right:new U("shift_right",F.token,">>"),less_than:new U("less_than",F.token,"<"),less_than_equal:new U("less_than_equal",F.token,"<="),shift_left:new U("shift_left",F.token,"<<"),modulo:new U("modulo",F.token,"%"),minus:new U("minus",F.token,"-"),minus_minus:new U("minus_minus",F.token,"--"),period:new U("period",F.token,"."),plus:new U("plus",F.token,"+"),plus_plus:new U("plus_plus",F.token,"++"),or:new U("or",F.token,"|"),or_or:new U("or_or",F.token,"||"),paren_left:new U("paren_left",F.token,"("),paren_right:new U("paren_right",F.token,")"),semicolon:new U("semicolon",F.token,";"),star:new U("star",F.token,"*"),tilde:new U("tilde",F.token,"~"),underscore:new U("underscore",F.token,"_"),xor:new U("xor",F.token,"^"),plus_equal:new U("plus_equal",F.token,"+="),minus_equal:new U("minus_equal",F.token,"-="),times_equal:new U("times_equal",F.token,"*="),division_equal:new U("division_equal",F.token,"/="),modulo_equal:new U("modulo_equal",F.token,"%="),and_equal:new U("and_equal",F.token,"&="),or_equal:new U("or_equal",F.token,"|="),xor_equal:new U("xor_equal",F.token,"^="),shift_right_equal:new U("shift_right_equal",F.token,">>="),shift_left_equal:new U("shift_left_equal",F.token,"<<=")},O.simpleTokens={"@":G.tokens.attr,"{":G.tokens.brace_left,"}":G.tokens.brace_right,":":G.tokens.colon,",":G.tokens.comma,"(":G.tokens.paren_left,")":G.tokens.paren_right,";":G.tokens.semicolon},O.literalTokens={"&":G.tokens.and,"&&":G.tokens.and_and,"->":G.tokens.arrow,"/":G.tokens.forward_slash,"!":G.tokens.bang,"[":G.tokens.bracket_left,"]":G.tokens.bracket_right,"=":G.tokens.equal,"==":G.tokens.equal_equal,"!=":G.tokens.not_equal,">":G.tokens.greater_than,">=":G.tokens.greater_than_equal,">>":G.tokens.shift_right,"<":G.tokens.less_than,"<=":G.tokens.less_than_equal,"<<":G.tokens.shift_left,"%":G.tokens.modulo,"-":G.tokens.minus,"--":G.tokens.minus_minus,".":G.tokens.period,"+":G.tokens.plus,"++":G.tokens.plus_plus,"|":G.tokens.or,"||":G.tokens.or_or,"*":G.tokens.star,"~":G.tokens.tilde,_:G.tokens.underscore,"^":G.tokens.xor,"+=":G.tokens.plus_equal,"-=":G.tokens.minus_equal,"*=":G.tokens.times_equal,"/=":G.tokens.division_equal,"%=":G.tokens.modulo_equal,"&=":G.tokens.and_equal,"|=":G.tokens.or_equal,"^=":G.tokens.xor_equal,">>=":G.tokens.shift_right_equal,"<<=":G.tokens.shift_left_equal},O.regexTokens={decimal_float_literal:G.tokens.decimal_float_literal,hex_float_literal:G.tokens.hex_float_literal,int_literal:G.tokens.int_literal,uint_literal:G.tokens.uint_literal,ident:G.tokens.ident},O.storage_class=[G.keywords.function,G.keywords.private,G.keywords.workgroup,G.keywords.uniform,G.keywords.storage],O.access_mode=[G.keywords.read,G.keywords.write,G.keywords.read_write],O.sampler_type=[G.keywords.sampler,G.keywords.sampler_comparison],O.sampled_texture_type=[G.keywords.texture_1d,G.keywords.texture_2d,G.keywords.texture_2d_array,G.keywords.texture_3d,G.keywords.texture_cube,G.keywords.texture_cube_array],O.multisampled_texture_type=[G.keywords.texture_multisampled_2d],O.storage_texture_type=[G.keywords.texture_storage_1d,G.keywords.texture_storage_2d,G.keywords.texture_storage_2d_array,G.keywords.texture_storage_3d],O.depth_texture_type=[G.keywords.texture_depth_2d,G.keywords.texture_depth_2d_array,G.keywords.texture_depth_cube,G.keywords.texture_depth_cube_array,G.keywords.texture_depth_multisampled_2d],O.texture_external_type=[G.keywords.texture_external],O.any_texture_type=[...G.sampled_texture_type,...G.multisampled_texture_type,...G.storage_texture_type,...G.depth_texture_type,...G.texture_external_type],O.texel_format=[G.keywords.r8unorm,G.keywords.r8snorm,G.keywords.r8uint,G.keywords.r8sint,G.keywords.r16uint,G.keywords.r16sint,G.keywords.r16float,G.keywords.rg8unorm,G.keywords.rg8snorm,G.keywords.rg8uint,G.keywords.rg8sint,G.keywords.r32uint,G.keywords.r32sint,G.keywords.r32float,G.keywords.rg16uint,G.keywords.rg16sint,G.keywords.rg16float,G.keywords.rgba8unorm,G.keywords.rgba8unorm_srgb,G.keywords.rgba8snorm,G.keywords.rgba8uint,G.keywords.rgba8sint,G.keywords.bgra8unorm,G.keywords.bgra8unorm_srgb,G.keywords.rgb10a2unorm,G.keywords.rg11b10float,G.keywords.rg32uint,G.keywords.rg32sint,G.keywords.rg32float,G.keywords.rgba16uint,G.keywords.rgba16sint,G.keywords.rgba16float,G.keywords.rgba32uint,G.keywords.rgba32sint,G.keywords.rgba32float],O.const_literal=[G.tokens.int_literal,G.tokens.uint_literal,G.tokens.decimal_float_literal,G.tokens.hex_float_literal,G.keywords.true,G.keywords.false],O.literal_or_ident=[G.tokens.ident,G.tokens.int_literal,G.tokens.uint_literal,G.tokens.decimal_float_literal,G.tokens.hex_float_literal,G.tokens.name],O.element_count_expression=[G.tokens.int_literal,G.tokens.uint_literal,G.tokens.ident],O.template_types=[G.keywords.vec2,G.keywords.vec3,G.keywords.vec4,G.keywords.mat2x2,G.keywords.mat2x3,G.keywords.mat2x4,G.keywords.mat3x2,G.keywords.mat3x3,G.keywords.mat3x4,G.keywords.mat4x2,G.keywords.mat4x3,G.keywords.mat4x4,G.keywords.atomic,G.keywords.bitcast,...G.any_texture_type],O.attribute_name=[G.tokens.ident,G.keywords.block,G.keywords.diagnostic],O.assignment_operators=[G.tokens.equal,G.tokens.plus_equal,G.tokens.minus_equal,G.tokens.times_equal,G.tokens.division_equal,G.tokens.modulo_equal,G.tokens.and_equal,G.tokens.or_equal,G.tokens.xor_equal,G.tokens.shift_right_equal,G.tokens.shift_left_equal],O.increment_operators=[G.tokens.plus_plus,G.tokens.minus_minus];class Ed{constructor(e,t,n,r,i){this.type=e,this.lexeme=t,this.line=n,this.start=r,this.end=i}toString(){return this.lexeme}isTemplateType(){return O.template_types.indexOf(this.type)!=-1}isArrayType(){return this.type==O.keywords.array}isArrayOrTemplateType(){return this.isArrayType()||this.isTemplateType()}}class LA{constructor(e){this._tokens=[],this._start=0,this._current=0,this._line=1,this._source=e??""}scanTokens(){for(;!this._isAtEnd();)if(this._start=this._current,!this.scanToken())throw"Invalid syntax at line "+this._line;return this._tokens.push(new Ed(O.eof,"",this._line,this._current,this._current)),this._tokens}scanToken(){let e=this._advance();if(e==`
`)return this._line++,!0;if(this._isWhitespace(e))return!0;if(e=="/"){if(this._peekAhead()=="/"){for(;e!=`
`;){if(this._isAtEnd())return!0;e=this._advance()}return this._line++,!0}if(this._peekAhead()=="*"){this._advance();let o=1;for(;o>0;){if(this._isAtEnd())return!0;if(e=this._advance(),e==`
`)this._line++;else if(e=="*"){if(this._peekAhead()=="/"&&(this._advance(),o--,o==0))return!0}else e=="/"&&this._peekAhead()=="*"&&(this._advance(),o++)}return!0}}const t=O.simpleTokens[e];if(t)return this._addToken(t),!0;let n=O.none;const r=this._isAlpha(e),i=e==="_";if(this._isAlphaNumeric(e)){let o=this._peekAhead();for(;this._isAlphaNumeric(o);)e+=this._advance(),o=this._peekAhead()}if(r){const o=O.keywords[e];if(o)return this._addToken(o),!0}if(r||i)return this._addToken(O.tokens.ident),!0;for(;;){let o=this._findType(e);const a=this._peekAhead();if(e=="-"&&this._tokens.length>0){if(a=="=")return this._current++,e+=a,this._addToken(O.tokens.minus_equal),!0;if(a=="-")return this._current++,e+=a,this._addToken(O.tokens.minus_minus),!0;const l=this._tokens.length-1;if((O.literal_or_ident.indexOf(this._tokens[l].type)!=-1||this._tokens[l].type==O.tokens.paren_right)&&a!=">")return this._addToken(o),!0}if(e==">"&&(a==">"||a=="=")){let l=!1,u=this._tokens.length-1;for(let c=0;c<5&&u>=0&&O.assignment_operators.indexOf(this._tokens[u].type)===-1;++c,--u)if(this._tokens[u].type===O.tokens.less_than){u>0&&this._tokens[u-1].isArrayOrTemplateType()&&(l=!0);break}if(l)return this._addToken(o),!0}if(o===O.none){let l=e,u=0;const c=2;for(let h=0;h<c;++h)if(l+=this._peekAhead(h),o=this._findType(l),o!==O.none){u=h;break}if(o===O.none)return n!==O.none&&(this._current--,this._addToken(n),!0);e=l,this._current+=u+1}if(n=o,this._isAtEnd())break;e+=this._advance()}return n!==O.none&&(this._addToken(n),!0)}_findType(e){for(const t in O.regexTokens){const n=O.regexTokens[t];if(this._match(e,n.rule))return n}return O.literalTokens[e]||O.none}_match(e,t){const n=t.exec(e);return n&&n.index==0&&n[0]==e}_isAtEnd(){return this._current>=this._source.length}_isAlpha(e){return!this._isNumeric(e)&&!this._isWhitespace(e)&&e!=="_"&&e!=="."&&e!=="("&&e!==")"&&e!=="["&&e!=="]"&&e!=="{"&&e!=="}"&&e!==","&&e!==";"&&e!==":"&&e!=="="&&e!=="!"&&e!=="<"&&e!==">"&&e!=="+"&&e!=="-"&&e!=="*"&&e!=="/"&&e!=="%"&&e!=="&"&&e!=="|"&&e!=="^"&&e!=="~"&&e!=="@"&&e!=="#"&&e!=="?"&&e!=="'"&&e!=="`"&&e!=='"'&&e!=="\\"&&e!==`
`&&e!=="\r"&&e!=="	"&&e!=="\0"}_isNumeric(e){return e>="0"&&e<="9"}_isAlphaNumeric(e){return this._isAlpha(e)||this._isNumeric(e)||e==="_"}_isWhitespace(e){return e==" "||e=="	"||e=="\r"}_advance(e=0){let t=this._source[this._current];return e=e||0,e++,this._current+=e,t}_peekAhead(e=0){return e=e||0,this._current+e>=this._source.length?"\0":this._source[this._current+e]}_addToken(e){const t=this._source.substring(this._start,this._current);this._tokens.push(new Ed(e,t,this._line,this._start,this._current))}}function re(s){return Array.isArray(s)||s?.buffer instanceof ArrayBuffer}const wl=new Float32Array(1),BA=new Uint32Array(wl.buffer),FA=new Uint32Array(wl.buffer),xl=new Int32Array(1),UA=new Float32Array(xl.buffer),zA=new Uint32Array(xl.buffer),_l=new Uint32Array(1),WA=new Float32Array(_l.buffer),GA=new Int32Array(_l.buffer);function Td(s,e,t){if(e===t)return s;if(e==="f32"){if(t==="i32"||t==="x32")return wl[0]=s,BA[0];if(t==="u32")return wl[0]=s,FA[0]}else if(e==="i32"||e==="x32"){if(t==="f32")return xl[0]=s,UA[0];if(t==="u32")return xl[0]=s,zA[0]}else if(e==="u32"){if(t==="f32")return _l[0]=s,WA[0];if(t==="i32"||t==="x32")return _l[0]=s,GA[0]}return console.error(`Unsupported cast from ${e} to ${t}`),s}class VA{constructor(e){this.resources=null,this.inUse=!1,this.info=null,this.node=e}}class ca{constructor(e,t){this.align=e,this.size=t}}class Zn{constructor(){this.uniforms=[],this.storage=[],this.textures=[],this.samplers=[],this.aliases=[],this.overrides=[],this.structs=[],this.entry=new AA,this.functions=[],this._types=new Map,this._functions=new Map}_isStorageTexture(e){return e.name=="texture_storage_1d"||e.name=="texture_storage_2d"||e.name=="texture_storage_2d_array"||e.name=="texture_storage_3d"}updateAST(e){for(const t of e)t instanceof Eo&&this._functions.set(t.name,new VA(t));for(const t of e)if(t instanceof cs){const n=this.getTypeInfo(t,null);n instanceof Ns&&this.structs.push(n)}for(const t of e)if(t instanceof _h)this.aliases.push(this._getAliasInfo(t));else if(t instanceof wh){const n=t,r=this._getAttributeNum(n.attributes,"id",0),i=n.type!=null?this.getTypeInfo(n.type,n.attributes):null;this.overrides.push(new IA(n.name,i,n.attributes,r))}else if(this._isUniformVar(t)){const n=t,r=this._getAttributeNum(n.attributes,"group",0),i=this._getAttributeNum(n.attributes,"binding",0),o=this.getTypeInfo(n.type,n.attributes),a=new ua(n.name,o,r,i,n.attributes,Is.Uniform,n.access);a.access||(a.access="read"),this.uniforms.push(a)}else if(this._isStorageVar(t)){const n=t,r=this._getAttributeNum(n.attributes,"group",0),i=this._getAttributeNum(n.attributes,"binding",0),o=this.getTypeInfo(n.type,n.attributes),a=this._isStorageTexture(o),l=new ua(n.name,o,r,i,n.attributes,a?Is.StorageTexture:Is.Storage,n.access);l.access||(l.access="read"),this.storage.push(l)}else if(this._isTextureVar(t)){const n=t,r=this._getAttributeNum(n.attributes,"group",0),i=this._getAttributeNum(n.attributes,"binding",0),o=this.getTypeInfo(n.type,n.attributes),a=this._isStorageTexture(o),l=new ua(n.name,o,r,i,n.attributes,a?Is.StorageTexture:Is.Texture,n.access);l.access||(l.access="read"),a?this.storage.push(l):this.textures.push(l)}else if(this._isSamplerVar(t)){const n=t,r=this._getAttributeNum(n.attributes,"group",0),i=this._getAttributeNum(n.attributes,"binding",0),o=this.getTypeInfo(n.type,n.attributes),a=new ua(n.name,o,r,i,n.attributes,Is.Sampler,n.access);this.samplers.push(a)}for(const t of e)if(t instanceof Eo){const n=this._getAttribute(t,"vertex"),r=this._getAttribute(t,"fragment"),i=this._getAttribute(t,"compute"),o=n||r||i,a=new TA(t.name,o?.name,t.attributes);a.attributes=t.attributes,a.startLine=t.startLine,a.endLine=t.endLine,this.functions.push(a),this._functions.get(t.name).info=a,o&&(this._functions.get(t.name).inUse=!0,a.inUse=!0,a.resources=this._findResources(t,!!o),a.inputs=this._getInputs(t.args),a.outputs=this._getOutputs(t.returnType),this.entry[o.name].push(a)),a.arguments=t.args.map(l=>new EA(l.name,this.getTypeInfo(l.type,l.attributes),l.attributes)),a.returnType=t.returnType?this.getTypeInfo(t.returnType,t.attributes):null}for(const t of this._functions.values())t.info&&(t.info.inUse=t.inUse,this._addCalls(t.node,t.info.calls));for(const t of this._functions.values())t.node.search(n=>{var r,i,o;if(n instanceof Ly){if(n.value)if(re(n.value))for(const a of n.value)for(const l of this.overrides)a===l.name&&((r=t.info)===null||r===void 0||r.overrides.push(l));else for(const a of this.overrides)n.value===a.name&&((i=t.info)===null||i===void 0||i.overrides.push(a))}else if(n instanceof Yt)for(const a of this.overrides)n.name===a.name&&((o=t.info)===null||o===void 0||o.overrides.push(a))});for(const t of this.uniforms)this._markStructsInUse(t.type);for(const t of this.storage)this._markStructsInUse(t.type)}getStructInfo(e){for(const t of this.structs)if(t.name==e)return t;return null}getOverrideInfo(e){for(const t of this.overrides)if(t.name==e)return t;return null}_markStructsInUse(e){if(e)if(e.isStruct){if(e.inUse=!0,e.members)for(const t of e.members)this._markStructsInUse(t.type)}else if(e.isArray)this._markStructsInUse(e.format);else if(e.isTemplate)e.format&&this._markStructsInUse(e.format);else{const t=this._getAlias(e.name);t&&this._markStructsInUse(t)}}_addCalls(e,t){var n;for(const r of e.calls){const i=(n=this._functions.get(r.name))===null||n===void 0?void 0:n.info;i&&t.add(i)}}findResource(e,t,n){if(n){for(const r of this.entry.compute)if(r.name===n){for(const i of r.resources)if(i.group==e&&i.binding==t)return i}for(const r of this.entry.vertex)if(r.name===n){for(const i of r.resources)if(i.group==e&&i.binding==t)return i}for(const r of this.entry.fragment)if(r.name===n){for(const i of r.resources)if(i.group==e&&i.binding==t)return i}}for(const r of this.uniforms)if(r.group==e&&r.binding==t)return r;for(const r of this.storage)if(r.group==e&&r.binding==t)return r;for(const r of this.textures)if(r.group==e&&r.binding==t)return r;for(const r of this.samplers)if(r.group==e&&r.binding==t)return r;return null}_findResource(e){for(const t of this.uniforms)if(t.name==e)return t;for(const t of this.storage)if(t.name==e)return t;for(const t of this.textures)if(t.name==e)return t;for(const t of this.samplers)if(t.name==e)return t;return null}_markStructsFromAST(e){const t=this.getTypeInfo(e,null);this._markStructsInUse(t)}_findResources(e,t){const n=[],r=this,i=[];return e.search(o=>{if(o instanceof yl)i.push({});else if(o instanceof bl)i.pop();else if(o instanceof ms){const a=o;t&&a.type!==null&&this._markStructsFromAST(a.type),i.length>0&&(i[i.length-1][a.name]=a)}else if(o instanceof Hn){const a=o;t&&a.type!==null&&this._markStructsFromAST(a.type)}else if(o instanceof oo){const a=o;t&&a.type!==null&&this._markStructsFromAST(a.type),i.length>0&&(i[i.length-1][a.name]=a)}else if(o instanceof Yt){const a=o;if(i.length>0&&i[i.length-1][a.name])return;const l=r._findResource(a.name);l&&n.push(l)}else if(o instanceof vh){const a=o,l=r._functions.get(a.name);l&&(t&&(l.inUse=!0),e.calls.add(l.node),l.resources===null&&(l.resources=r._findResources(l.node,t)),n.push(...l.resources))}else if(o instanceof xh){const a=o,l=r._functions.get(a.name);l&&(t&&(l.inUse=!0),e.calls.add(l.node),l.resources===null&&(l.resources=r._findResources(l.node,t)),n.push(...l.resources))}}),[...new Map(n.map(o=>[o.name,o])).values()]}getBindGroups(){const e=[];function t(n,r){n>=e.length&&(e.length=n+1),e[n]===void 0&&(e[n]=[]),r>=e[n].length&&(e[n].length=r+1)}for(const n of this.uniforms)t(n.group,n.binding),e[n.group][n.binding]=n;for(const n of this.storage)t(n.group,n.binding),e[n.group][n.binding]=n;for(const n of this.textures)t(n.group,n.binding),e[n.group][n.binding]=n;for(const n of this.samplers)t(n.group,n.binding),e[n.group][n.binding]=n;return e}_getOutputs(e,t=void 0){if(t===void 0&&(t=[]),e instanceof cs)this._getStructOutputs(e,t);else{const n=this._getOutputInfo(e);n!==null&&t.push(n)}return t}_getStructOutputs(e,t){for(const n of e.members)if(n.type instanceof cs)this._getStructOutputs(n.type,t);else{const r=this._getAttribute(n,"location")||this._getAttribute(n,"builtin");if(r!==null){const i=this.getTypeInfo(n.type,n.type.attributes),o=this._parseInt(r.value),a=new _d(n.name,i,r.name,o);t.push(a)}}}_getOutputInfo(e){const t=this._getAttribute(e,"location")||this._getAttribute(e,"builtin");if(t!==null){const n=this.getTypeInfo(e,e.attributes),r=this._parseInt(t.value);return new _d("",n,t.name,r)}return null}_getInputs(e,t=void 0){t===void 0&&(t=[]);for(const n of e)if(n.type instanceof cs)this._getStructInputs(n.type,t);else{const r=this._getInputInfo(n);r!==null&&t.push(r)}return t}_getStructInputs(e,t){for(const n of e.members)if(n.type instanceof cs)this._getStructInputs(n.type,t);else{const r=this._getInputInfo(n);r!==null&&t.push(r)}}_getInputInfo(e){const t=this._getAttribute(e,"location")||this._getAttribute(e,"builtin");if(t!==null){const n=this._getAttribute(e,"interpolation"),r=this.getTypeInfo(e.type,e.attributes),i=this._parseInt(t.value),o=new kA(e.name,r,t.name,i);return n!==null&&(o.interpolation=this._parseString(n.value)),o}return null}_parseString(e){return e instanceof Array&&(e=e[0]),e}_parseInt(e){e instanceof Array&&(e=e[0]);const t=parseInt(e);return isNaN(t)?e:t}_getAlias(e){for(const t of this.aliases)if(t.name==e)return t.type;return null}_getAliasInfo(e){return new SA(e.name,this.getTypeInfo(e.type,null))}getTypeInfoByName(e){for(const t of this.structs)if(t.name==e)return t;for(const t of this.aliases)if(t.name==e)return t.type;return null}getTypeInfo(e,t=null){if(this._types.has(e))return this._types.get(e);if(e instanceof Ta){const r=e.type?this.getTypeInfo(e.type,e.attributes):null,i=new tc(e.name,r,t);return this._types.set(e,i),this._updateTypeInfo(i),i}if(e instanceof ao){const r=e,i=r.format?this.getTypeInfo(r.format,r.attributes):null,o=new Ps(r.name,t);return o.format=i,o.count=r.count,this._types.set(e,o),this._updateTypeInfo(o),o}if(e instanceof cs){const r=e,i=new Ns(r.name,t);i.startLine=r.startLine,i.endLine=r.endLine;for(const o of r.members){const a=this.getTypeInfo(o.type,o.attributes);i.members.push(new xd(o.name,a,o.attributes))}return this._types.set(e,i),this._updateTypeInfo(i),i}if(e instanceof qi){const r=e,i=r.format instanceof Y,o=r.format?i?this.getTypeInfo(r.format,null):new an(r.format,null):null,a=new Ar(r.name,o,t,r.access);return this._types.set(e,a),this._updateTypeInfo(a),a}if(e instanceof z){const r=e,i=r.format?this.getTypeInfo(r.format,null):null,o=new Ar(r.name,i,t,r.access);return this._types.set(e,o),this._updateTypeInfo(o),o}const n=new an(e.name,t);return this._types.set(e,n),this._updateTypeInfo(n),n}_updateTypeInfo(e){var t,n,r;const i=this._getTypeSize(e);if(e.size=(t=i?.size)!==null&&t!==void 0?t:0,e instanceof Ps&&e.format){const o=this._getTypeSize(e.format);e.stride=Math.max((n=o?.size)!==null&&n!==void 0?n:0,(r=o?.align)!==null&&r!==void 0?r:0),this._updateTypeInfo(e.format)}e instanceof tc&&this._updateTypeInfo(e.format),e instanceof Ns&&this._updateStructInfo(e)}_updateStructInfo(e){var t;let n=0,r=0,i=0,o=0;for(let a=0,l=e.members.length;a<l;++a){const u=e.members[a],c=this._getTypeSize(u);if(!c)continue;(t=this._getAlias(u.type.name))!==null&&t!==void 0||u.type;const h=c.align,d=c.size;n=this._roundUp(h,n+r),r=d,i=n,o=Math.max(o,h),u.offset=n,u.size=d,this._updateTypeInfo(u.type)}e.size=this._roundUp(o,i+r),e.align=o}_getTypeSize(e){var t,n;if(e==null)return null;const r=this._getAttributeNum(e.attributes,"size",0),i=this._getAttributeNum(e.attributes,"align",0);if(e instanceof xd&&(e=e.type),e instanceof an){const o=this._getAlias(e.name);o!==null&&(e=o)}{const o=Zn._typeInfo[e.name];if(o!==void 0){const a=((t=e.format)===null||t===void 0?void 0:t.name)==="f16"?2:1;return new ca(Math.max(i,o.align/a),Math.max(r,o.size/a))}}{const o=Zn._typeInfo[e.name.substring(0,e.name.length-1)];if(o){const a=e.name[e.name.length-1]==="h"?2:1;return new ca(Math.max(i,o.align/a),Math.max(r,o.size/a))}}if(e instanceof Ps){let o=e,a=8,l=8;const u=this._getTypeSize(o.format);return u!==null&&(l=u.size,a=u.align),l=o.count*this._getAttributeNum((n=e?.attributes)!==null&&n!==void 0?n:null,"stride",this._roundUp(a,l)),r&&(l=r),new ca(Math.max(i,a),Math.max(r,l))}if(e instanceof Ns){let o=0,a=0,l=0,u=0,c=0;for(const h of e.members){const d=this._getTypeSize(h.type);d!==null&&(o=Math.max(d.align,o),l=this._roundUp(d.align,l+u),u=d.size,c=l)}return a=this._roundUp(o,c+u),new ca(Math.max(i,o),Math.max(r,a))}return null}_isUniformVar(e){return e instanceof ms&&e.storage=="uniform"}_isStorageVar(e){return e instanceof ms&&e.storage=="storage"}_isTextureVar(e){return e instanceof ms&&e.type!==null&&Zn._textureTypes.indexOf(e.type.name)!=-1}_isSamplerVar(e){return e instanceof ms&&e.type!==null&&Zn._samplerTypes.indexOf(e.type.name)!=-1}_getAttribute(e,t){const n=e;if(!n||!n.attributes)return null;const r=n.attributes;for(let i of r)if(i.name==t)return i;return null}_getAttributeNum(e,t,n){if(e===null)return n;for(let r of e)if(r.name==t){let i=r!==null&&r.value!==null?r.value:n;return i instanceof Array&&(i=i[0]),typeof i=="number"?i:typeof i=="string"?parseInt(i):n}return n}_roundUp(e,t){return Math.ceil(t/e)*e}}Zn._typeInfo={f16:{align:2,size:2},i32:{align:4,size:4},u32:{align:4,size:4},f32:{align:4,size:4},atomic:{align:4,size:4},vec2:{align:8,size:8},vec3:{align:16,size:12},vec4:{align:16,size:16},mat2x2:{align:8,size:16},mat3x2:{align:8,size:24},mat4x2:{align:8,size:32},mat2x3:{align:16,size:32},mat3x3:{align:16,size:48},mat4x3:{align:16,size:64},mat2x4:{align:16,size:32},mat3x4:{align:16,size:48},mat4x4:{align:16,size:64}},Zn._textureTypes=O.any_texture_type.map(s=>s.name),Zn._samplerTypes=O.sampler_type.map(s=>s.name);let Sh=0;class kh{constructor(e,t,n){this.id=Sh++,this.name=e,this.value=t,this.node=n}clone(){return new kh(this.name,this.value,this.node)}}class Ih{constructor(e){this.id=Sh++,this.name=e.name,this.node=e}clone(){return new Ih(this.node)}}class Eh{constructor(e){this.parent=null,this.variables=new Map,this.functions=new Map,this.currentFunctionName="",this.id=Sh++,e&&(this.parent=e,this.currentFunctionName=e.currentFunctionName)}getVariable(e){var t;return this.variables.has(e)?(t=this.variables.get(e))!==null&&t!==void 0?t:null:this.parent?this.parent.getVariable(e):null}getFunction(e){var t;return this.functions.has(e)?(t=this.functions.get(e))!==null&&t!==void 0?t:null:this.parent?this.parent.getFunction(e):null}createVariable(e,t,n){this.variables.set(e,new kh(e,t,n??null))}setVariable(e,t,n){const r=this.getVariable(e);r!==null?r.value=t:this.createVariable(e,t,n)}getVariableValue(e){var t;return(t=this.getVariable(e)?.value)!==null&&t!==void 0?t:null}clone(){return new Eh(this)}}class qA{evalExpression(e,t){return null}getTypeInfo(e){return null}getVariableName(e,t){return""}}class HA{constructor(e){this.exec=e}getTypeInfo(e){return this.exec.getTypeInfo(e)}All(e,t){const n=this.exec.evalExpression(e.args[0],t);let r=!0;if(n instanceof P)return n.data.forEach(i=>{i||(r=!1)}),new B(r?1:0,this.getTypeInfo("bool"));throw Error("All() expects a vector argument. Line "+e.line)}Any(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof P){const r=n.data.some(i=>i);return new B(r?1:0,this.getTypeInfo("bool"))}throw Error("Any() expects a vector argument. Line "+e.line)}Select(e,t){const n=this.exec.evalExpression(e.args[2],t);if(!(n instanceof B))throw Error("Select() expects a bool condition. Line "+e.line);return n.value?this.exec.evalExpression(e.args[1],t):this.exec.evalExpression(e.args[0],t)}ArrayLength(e,t){let n=e.args[0];n instanceof Xe&&(n=n.right);const r=this.exec.evalExpression(n,t);if(r instanceof qe&&r.typeInfo.size===0){const i=r.typeInfo,o=r.buffer.byteLength/i.stride;return new B(o,this.getTypeInfo("u32"))}return new B(r.typeInfo.size,this.getTypeInfo("u32"))}Abs(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof P)return new P(n.data.map(i=>Math.abs(i)),n.typeInfo);const r=n;return new B(Math.abs(r.value),r.typeInfo)}Acos(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.acos(r)),n.typeInfo):new B(Math.acos(n.value),n.typeInfo)}Acosh(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.acosh(r)),n.typeInfo):new B(Math.acosh(n.value),n.typeInfo)}Asin(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.asin(r)),n.typeInfo):new B(Math.asin(n.value),n.typeInfo)}Asinh(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.asinh(r)),n.typeInfo):new B(Math.asinh(n.value),n.typeInfo)}Atan(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.atan(r)),n.typeInfo):new B(Math.atan(n.value),n.typeInfo)}Atanh(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.atanh(r)),n.typeInfo):new B(Math.atanh(n.value),n.typeInfo)}Atan2(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);if(n instanceof P&&r instanceof P)return new P(n.data.map((o,a)=>Math.atan2(o,r.data[a])),n.typeInfo);const i=r;return new B(Math.atan2(n.value,i.value),n.typeInfo)}Ceil(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.ceil(r)),n.typeInfo):new B(Math.ceil(n.value),n.typeInfo)}_clamp(e,t,n){return Math.min(Math.max(e,t),n)}Clamp(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t),i=this.exec.evalExpression(e.args[2],t);if(n instanceof P&&r instanceof P&&i instanceof P)return new P(n.data.map((u,c)=>this._clamp(u,r.data[c],i.data[c])),n.typeInfo);const o=n,a=r,l=i;return new B(this._clamp(o.value,a.value,l.value),n.typeInfo)}Cos(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.cos(r)),n.typeInfo):new B(Math.cos(n.value),n.typeInfo)}Cosh(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.cosh(r)),n.typeInfo):new B(Math.cos(n.value),n.typeInfo)}CountLeadingZeros(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.clz32(r)),n.typeInfo):new B(Math.clz32(n.value),n.typeInfo)}_countOneBits(e){let t=0;for(;e!==0;)1&e&&t++,e>>=1;return t}CountOneBits(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof P)return new P(n.data.map(i=>this._countOneBits(i)),n.typeInfo);const r=n;return new B(this._countOneBits(r.value),n.typeInfo)}_countTrailingZeros(e){if(e===0)return 32;let t=0;for(;!(1&e);)e>>=1,t++;return t}CountTrailingZeros(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof P)return new P(n.data.map(i=>this._countTrailingZeros(i)),n.typeInfo);const r=n;return new B(this._countTrailingZeros(r.value),n.typeInfo)}Cross(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);if(n instanceof P&&r instanceof P){if(n.data.length!==3||r.data.length!==3)return console.error("Cross() expects 3D vectors. Line "+e.line),null;const i=n.data,o=r.data;return new P([i[1]*o[2]-o[1]*i[2],i[2]*o[0]-o[2]*i[0],i[0]*o[1]-o[0]*i[1]],n.typeInfo)}return console.error("Cross() expects vector arguments. Line "+e.line),null}Degrees(e,t){const n=this.exec.evalExpression(e.args[0],t),r=180/Math.PI;return n instanceof P?new P(n.data.map(i=>i*r),n.typeInfo):new B(n.value*r,this.getTypeInfo("f32"))}Determinant(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof de){const r=n.data,i=n.typeInfo.getTypeName(),o=i.endsWith("h")?this.getTypeInfo("f16"):this.getTypeInfo("f32");if(i==="mat2x2"||i==="mat2x2f"||i==="mat2x2h")return new B(r[0]*r[3]-r[1]*r[2],o);if(i==="mat2x3"||i==="mat2x3f"||i==="mat2x3h")return new B(r[0]*(r[4]*r[8]-r[5]*r[7])-r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6]),o);if(i==="mat2x4"||i==="mat2x4f"||i==="mat2x4h")console.error("TODO: Determinant for "+i);else if(i==="mat3x2"||i==="mat3x2f"||i==="mat3x2h")console.error("TODO: Determinant for "+i);else{if(i==="mat3x3"||i==="mat3x3f"||i==="mat3x3h")return new B(r[0]*(r[4]*r[8]-r[5]*r[7])-r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6]),o);i==="mat3x4"||i==="mat3x4f"||i==="mat3x4h"||i==="mat4x2"||i==="mat4x2f"||i==="mat4x2h"||i==="mat4x3"||i==="mat4x3f"||i==="mat4x3h"?console.error("TODO: Determinant for "+i):i!=="mat4x4"&&i!=="mat4x4f"&&i!=="mat4x4h"||console.error("TODO: Determinant for "+i)}}return console.error("Determinant expects a matrix argument. Line "+e.line),null}Distance(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);if(n instanceof P&&r instanceof P){let o=0;for(let a=0;a<n.data.length;++a)o+=(n.data[a]-r.data[a])*(n.data[a]-r.data[a]);return new B(Math.sqrt(o),this.getTypeInfo("f32"))}const i=r;return new B(Math.abs(n.value-i.value),n.typeInfo)}_dot(e,t){let n=0;for(let r=0;r<e.length;++r)n+=t[r]*e[r];return n}Dot(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);return n instanceof P&&r instanceof P?new B(this._dot(n.data,r.data),this.getTypeInfo("f32")):(console.error("Dot() expects vector arguments. Line "+e.line),null)}Dot4U8Packed(e,t){return console.error("TODO: dot4U8Packed. Line "+e.line),null}Dot4I8Packed(e,t){return console.error("TODO: dot4I8Packed. Line "+e.line),null}Exp(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.exp(r)),n.typeInfo):new B(Math.exp(n.value),n.typeInfo)}Exp2(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.pow(2,r)),n.typeInfo):new B(Math.pow(2,n.value),n.typeInfo)}ExtractBits(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t),i=this.exec.evalExpression(e.args[2],t);if(r.typeInfo.name!=="u32"&&r.typeInfo.name!=="x32")return console.error("ExtractBits() expects an i32 offset argument. Line "+e.line),null;if(i.typeInfo.name!=="u32"&&i.typeInfo.name!=="x32")return console.error("ExtractBits() expects an i32 count argument. Line "+e.line),null;const o=r.value,a=i.value;if(n instanceof P)return new P(n.data.map(u=>u>>o&(1<<a)-1),n.typeInfo);if(n.typeInfo.name!=="i32"&&n.typeInfo.name!=="x32")return console.error("ExtractBits() expects an i32 argument. Line "+e.line),null;const l=n.value;return new B(l>>o&(1<<a)-1,this.getTypeInfo("i32"))}FaceForward(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t),i=this.exec.evalExpression(e.args[2],t);if(n instanceof P&&r instanceof P&&i instanceof P){const o=this._dot(r.data,i.data);return new P(o<0?Array.from(n.data):n.data.map(a=>-a),n.typeInfo)}return console.error("FaceForward() expects vector arguments. Line "+e.line),null}_firstLeadingBit(e){return e===0?-1:31-Math.clz32(e)}FirstLeadingBit(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof P)return new P(n.data.map(i=>this._firstLeadingBit(i)),n.typeInfo);const r=n;return new B(this._firstLeadingBit(r.value),n.typeInfo)}_firstTrailingBit(e){return e===0?-1:Math.log2(e&-e)}FirstTrailingBit(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof P)return new P(n.data.map(i=>this._firstTrailingBit(i)),n.typeInfo);const r=n;return new B(this._firstTrailingBit(r.value),n.typeInfo)}Floor(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.floor(r)),n.typeInfo):new B(Math.floor(n.value),n.typeInfo)}Fma(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t),i=this.exec.evalExpression(e.args[2],t);if(n instanceof P&&r instanceof P&&i instanceof P)return n.data.length!==r.data.length||n.data.length!==i.data.length?(console.error("Fma() expects vectors of the same length. Line "+e.line),null):new P(n.data.map((u,c)=>u*r.data[c]+i.data[c]),n.typeInfo);const o=n,a=r,l=i;return new B(o.value*a.value+l.value,o.typeInfo)}Fract(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof P)return new P(n.data.map(i=>i-Math.floor(i)),n.typeInfo);const r=n;return new B(r.value-Math.floor(r.value),n.typeInfo)}Frexp(e,t){return console.error("TODO: frexp. Line "+e.line),null}InsertBits(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t),i=this.exec.evalExpression(e.args[2],t),o=this.exec.evalExpression(e.args[3],t);if(i.typeInfo.name!=="u32"&&i.typeInfo.name!=="x32")return console.error("InsertBits() expects an i32 offset argument. Line "+e.line),null;const a=i.value,l=(1<<o.value)-1<<a,u=~l;if(n instanceof P&&r instanceof P)return new P(n.data.map((d,w)=>d&u|r.data[w]<<a&l),n.typeInfo);const c=n.value,h=r.value;return new B(c&u|h<<a&l,n.typeInfo)}InverseSqrt(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>1/Math.sqrt(r)),n.typeInfo):new B(1/Math.sqrt(n.value),n.typeInfo)}Ldexp(e,t){return console.error("TODO: ldexp. Line "+e.line),null}Length(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof P){let r=0;return n.data.forEach(i=>{r+=i*i}),new B(Math.sqrt(r),this.getTypeInfo("f32"))}return new B(Math.abs(n.value),n.typeInfo)}Log(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.log(r)),n.typeInfo):new B(Math.log(n.value),n.typeInfo)}Log2(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.log2(r)),n.typeInfo):new B(Math.log2(n.value),n.typeInfo)}Max(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);if(n instanceof P&&r instanceof P)return new P(n.data.map((o,a)=>Math.max(o,r.data[a])),n.typeInfo);const i=r;return new B(Math.max(n.value,i.value),n.typeInfo)}Min(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);if(n instanceof P&&r instanceof P)return new P(n.data.map((o,a)=>Math.min(o,r.data[a])),n.typeInfo);const i=r;return new B(Math.min(n.value,i.value),n.typeInfo)}Mix(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t),i=this.exec.evalExpression(e.args[2],t);if(n instanceof P&&r instanceof P&&i instanceof P)return new P(n.data.map((l,u)=>n.data[u]*(1-i.data[u])+r.data[u]*i.data[u]),n.typeInfo);const o=r,a=i;return new B(n.value*(1-a.value)+o.value*a.value,n.typeInfo)}Modf(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);if(n instanceof P&&r instanceof P)return new P(n.data.map((o,a)=>o%r.data[a]),n.typeInfo);const i=r;return new B(n.value%i.value,n.typeInfo)}Normalize(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof P){const r=this.Length(e,t).value;return new P(n.data.map(i=>i/r),n.typeInfo)}return console.error("Normalize() expects a vector argument. Line "+e.line),null}Pow(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);if(n instanceof P&&r instanceof P)return new P(n.data.map((o,a)=>Math.pow(o,r.data[a])),n.typeInfo);const i=r;return new B(Math.pow(n.value,i.value),n.typeInfo)}QuantizeToF16(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>r),n.typeInfo):new B(n.value,n.typeInfo)}Radians(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>r*Math.PI/180),n.typeInfo):new B(n.value*Math.PI/180,this.getTypeInfo("f32"))}Reflect(e,t){let n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);if(n instanceof P&&r instanceof P){const i=this._dot(n.data,r.data);return new P(n.data.map((o,a)=>o-2*i*r.data[a]),n.typeInfo)}return console.error("Reflect() expects vector arguments. Line "+e.line),null}Refract(e,t){let n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t),i=this.exec.evalExpression(e.args[2],t);if(n instanceof P&&r instanceof P&&i instanceof B){const o=this._dot(r.data,n.data);return new P(n.data.map((a,l)=>{const u=1-i.value*i.value*(1-o*o);if(u<0)return 0;const c=Math.sqrt(u);return i.value*a-(i.value*o+c)*r.data[l]}),n.typeInfo)}return console.error("Refract() expects vector arguments and a scalar argument. Line "+e.line),null}ReverseBits(e,t){return console.error("TODO: reverseBits. Line "+e.line),null}Round(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.round(r)),n.typeInfo):new B(Math.round(n.value),n.typeInfo)}Saturate(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.min(Math.max(r,0),1)),n.typeInfo):new B(Math.min(Math.max(n.value,0),1),n.typeInfo)}Sign(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.sign(r)),n.typeInfo):new B(Math.sign(n.value),n.typeInfo)}Sin(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.sin(r)),n.typeInfo):new B(Math.sin(n.value),n.typeInfo)}Sinh(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.sinh(r)),n.typeInfo):new B(Math.sinh(n.value),n.typeInfo)}_smoothstep(e,t,n){const r=Math.min(Math.max((n-e)/(t-e),0),1);return r*r*(3-2*r)}SmoothStep(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t),i=this.exec.evalExpression(e.args[2],t);if(i instanceof P&&n instanceof P&&r instanceof P)return new P(i.data.map((u,c)=>this._smoothstep(n.data[c],r.data[c],u)),i.typeInfo);const o=n,a=r,l=i;return new B(this._smoothstep(o.value,a.value,l.value),i.typeInfo)}Sqrt(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.sqrt(r)),n.typeInfo):new B(Math.sqrt(n.value),n.typeInfo)}Step(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);if(r instanceof P&&n instanceof P)return new P(r.data.map((o,a)=>o<n.data[a]?0:1),r.typeInfo);const i=n;return new B(r.value<i.value?0:1,i.typeInfo)}Tan(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.tan(r)),n.typeInfo):new B(Math.tan(n.value),n.typeInfo)}Tanh(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.tanh(r)),n.typeInfo):new B(Math.tanh(n.value),n.typeInfo)}_getTransposeType(e){const t=e.getTypeName();return t==="mat2x2f"||t==="mat2x2h"?e:t==="mat2x3f"?this.getTypeInfo("mat3x2f"):t==="mat2x3h"?this.getTypeInfo("mat3x2h"):t==="mat2x4f"?this.getTypeInfo("mat4x2f"):t==="mat2x4h"?this.getTypeInfo("mat4x2h"):t==="mat3x2f"?this.getTypeInfo("mat2x3f"):t==="mat3x2h"?this.getTypeInfo("mat2x3h"):t==="mat3x3f"||t==="mat3x3h"?e:t==="mat3x4f"?this.getTypeInfo("mat4x3f"):t==="mat3x4h"?this.getTypeInfo("mat4x3h"):t==="mat4x2f"?this.getTypeInfo("mat2x4f"):t==="mat4x2h"?this.getTypeInfo("mat2x4h"):t==="mat4x3f"?this.getTypeInfo("mat3x4f"):t==="mat4x3h"?this.getTypeInfo("mat3x4h"):(t==="mat4x4f"||t==="mat4x4h"||console.error("Invalid matrix type "+t),e)}Transpose(e,t){const n=this.exec.evalExpression(e.args[0],t);if(!(n instanceof de))return console.error("Transpose() expects a matrix argument. Line "+e.line),null;const r=this._getTransposeType(n.typeInfo);if(n.typeInfo.name==="mat2x2"||n.typeInfo.name==="mat2x2f"||n.typeInfo.name==="mat2x2h"){const i=n.data;return new de([i[0],i[2],i[1],i[3]],r)}if(n.typeInfo.name==="mat2x3"||n.typeInfo.name==="mat2x3f"||n.typeInfo.name==="mat2x3h"){const i=n.data;return new de([i[0],i[3],i[6],i[1],i[4],i[7]],r)}if(n.typeInfo.name==="mat2x4"||n.typeInfo.name==="mat2x4f"||n.typeInfo.name==="mat2x4h"){const i=n.data;return new de([i[0],i[4],i[8],i[12],i[1],i[5],i[9],i[13]],r)}if(n.typeInfo.name==="mat3x2"||n.typeInfo.name==="mat3x2f"||n.typeInfo.name==="mat3x2h"){const i=n.data;return new de([i[0],i[3],i[1],i[4],i[2],i[5]],r)}if(n.typeInfo.name==="mat3x3"||n.typeInfo.name==="mat3x3f"||n.typeInfo.name==="mat3x3h"){const i=n.data;return new de([i[0],i[3],i[6],i[1],i[4],i[7],i[2],i[5],i[8]],r)}if(n.typeInfo.name==="mat3x4"||n.typeInfo.name==="mat3x4f"||n.typeInfo.name==="mat3x4h"){const i=n.data;return new de([i[0],i[4],i[8],i[12],i[1],i[5],i[9],i[13],i[2],i[6],i[10],i[14]],r)}if(n.typeInfo.name==="mat4x2"||n.typeInfo.name==="mat4x2f"||n.typeInfo.name==="mat4x2h"){const i=n.data;return new de([i[0],i[4],i[1],i[5],i[2],i[6]],r)}if(n.typeInfo.name==="mat4x3"||n.typeInfo.name==="mat4x3f"||n.typeInfo.name==="mat4x3h"){const i=n.data;return new de([i[0],i[4],i[8],i[1],i[5],i[9],i[2],i[6],i[10]],r)}if(n.typeInfo.name==="mat4x4"||n.typeInfo.name==="mat4x4f"||n.typeInfo.name==="mat4x4h"){const i=n.data;return new de([i[0],i[4],i[8],i[12],i[1],i[5],i[9],i[13],i[2],i[6],i[10],i[14],i[3],i[7],i[11],i[15]],r)}return console.error("Invalid matrix type "+n.typeInfo.name),null}Trunc(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof P?new P(n.data.map(r=>Math.trunc(r)),n.typeInfo):new B(Math.trunc(n.value),n.typeInfo)}Dpdx(e,t){return console.error("TODO: dpdx. Line "+e.line),null}DpdxCoarse(e,t){return console.error("TODO: dpdxCoarse. Line "+e.line),null}DpdxFine(e,t){return console.error("TODO: dpdxFine"),null}Dpdy(e,t){return console.error("TODO: dpdy"),null}DpdyCoarse(e,t){return console.error("TODO: dpdyCoarse"),null}DpdyFine(e,t){return console.error("TODO: dpdyFine"),null}Fwidth(e,t){return console.error("TODO: fwidth"),null}FwidthCoarse(e,t){return console.error("TODO: fwidthCoarse"),null}FwidthFine(e,t){return console.error("TODO: fwidthFine"),null}TextureDimensions(e,t){const n=e.args[0],r=e.args.length>1?this.exec.evalExpression(e.args[1],t).value:0;if(n instanceof Yt){const i=n.name,o=t.getVariableValue(i);if(o instanceof hs){if(r<0||r>=o.mipLevelCount)return console.error("Invalid mip level for textureDimensions. Line "+e.line),null;const a=o.getMipLevelSize(r),l=o.dimension;return l==="1d"?new B(a[0],this.getTypeInfo("u32")):l==="3d"?new P(a,this.getTypeInfo("vec3u")):l==="2d"?new P(a.slice(0,2),this.getTypeInfo("vec2u")):(console.error(`Invalid texture dimension ${l} not found. Line ${e.line}`),null)}return console.error(`Texture ${i} not found. Line ${e.line}`),null}return console.error("Invalid texture argument for textureDimensions. Line "+e.line),null}TextureGather(e,t){return console.error("TODO: textureGather"),null}TextureGatherCompare(e,t){return console.error("TODO: textureGatherCompare"),null}TextureLoad(e,t){const n=e.args[0],r=this.exec.evalExpression(e.args[1],t),i=e.args.length>2?this.exec.evalExpression(e.args[2],t).value:0;if(!(r instanceof P)||r.data.length!==2)return console.error("Invalid UV argument for textureLoad. Line "+e.line),null;if(n instanceof Yt){const o=n.name,a=t.getVariableValue(o);if(a instanceof hs){const l=Math.floor(r.data[0]),u=Math.floor(r.data[1]);if(l<0||l>=a.width||u<0||u>=a.height)return console.error(`Texture ${o} out of bounds. Line ${e.line}`),null;const c=a.getPixel(l,u,0,i);return c===null?(console.error("Invalid texture format for textureLoad. Line "+e.line),null):new P(c,this.getTypeInfo("vec4f"))}return console.error(`Texture ${o} not found. Line ${e.line}`),null}return console.error("Invalid texture argument for textureLoad. Line "+e.line),null}TextureNumLayers(e,t){const n=e.args[0];if(n instanceof Yt){const r=n.name,i=t.getVariableValue(r);return i instanceof hs?new B(i.depthOrArrayLayers,this.getTypeInfo("u32")):(console.error(`Texture ${r} not found. Line ${e.line}`),null)}return console.error("Invalid texture argument for textureNumLayers. Line "+e.line),null}TextureNumLevels(e,t){const n=e.args[0];if(n instanceof Yt){const r=n.name,i=t.getVariableValue(r);return i instanceof hs?new B(i.mipLevelCount,this.getTypeInfo("u32")):(console.error(`Texture ${r} not found. Line ${e.line}`),null)}return console.error("Invalid texture argument for textureNumLevels. Line "+e.line),null}TextureNumSamples(e,t){const n=e.args[0];if(n instanceof Yt){const r=n.name,i=t.getVariableValue(r);return i instanceof hs?new B(i.sampleCount,this.getTypeInfo("u32")):(console.error(`Texture ${r} not found. Line ${e.line}`),null)}return console.error("Invalid texture argument for textureNumSamples. Line "+e.line),null}TextureSample(e,t){return console.error("TODO: textureSample"),null}TextureSampleBias(e,t){return console.error("TODO: textureSampleBias"),null}TextureSampleCompare(e,t){return console.error("TODO: textureSampleCompare"),null}TextureSampleCompareLevel(e,t){return console.error("TODO: textureSampleCompareLevel"),null}TextureSampleGrad(e,t){return console.error("TODO: textureSampleGrad"),null}TextureSampleLevel(e,t){return console.error("TODO: textureSampleLevel"),null}TextureSampleBaseClampToEdge(e,t){return console.error("TODO: textureSampleBaseClampToEdge"),null}TextureStore(e,t){const n=e.args[0],r=this.exec.evalExpression(e.args[1],t),i=e.args.length===4?this.exec.evalExpression(e.args[2],t).value:0,o=e.args.length===4?this.exec.evalExpression(e.args[3],t).data:this.exec.evalExpression(e.args[2],t).data;if(o.length!==4)return console.error("Invalid value argument for textureStore. Line "+e.line),null;if(!(r instanceof P)||r.data.length!==2)return console.error("Invalid UV argument for textureStore. Line "+e.line),null;if(n instanceof Yt){const a=n.name,l=t.getVariableValue(a);if(l instanceof hs){const u=l.getMipLevelSize(0),c=Math.floor(r.data[0]),h=Math.floor(r.data[1]);return c<0||c>=u[0]||h<0||h>=u[1]?(console.error(`Texture ${a} out of bounds. Line ${e.line}`),null):(l.setPixel(c,h,0,i,Array.from(o)),null)}return console.error(`Texture ${a} not found. Line ${e.line}`),null}return console.error("Invalid texture argument for textureStore. Line "+e.line),null}AtomicLoad(e,t){let n=e.args[0];n instanceof Xe&&(n=n.right);const r=this.exec.getVariableName(n,t);return t.getVariable(r).value.getSubData(this.exec,n.postfix,t)}AtomicStore(e,t){let n=e.args[0];n instanceof Xe&&(n=n.right);const r=this.exec.getVariableName(n,t),i=t.getVariable(r);let o=e.args[1];const a=this.exec.evalExpression(o,t),l=i.value.getSubData(this.exec,n.postfix,t);return l instanceof B&&a instanceof B&&(l.value=a.value),i.value instanceof qe&&i.value.setDataValue(this.exec,l,n.postfix,t),null}AtomicAdd(e,t){let n=e.args[0];n instanceof Xe&&(n=n.right);const r=this.exec.getVariableName(n,t),i=t.getVariable(r);let o=e.args[1];const a=this.exec.evalExpression(o,t),l=i.value.getSubData(this.exec,n.postfix,t),u=new B(l.value,l.typeInfo);return l instanceof B&&a instanceof B&&(l.value+=a.value),i.value instanceof qe&&i.value.setDataValue(this.exec,l,n.postfix,t),u}AtomicSub(e,t){let n=e.args[0];n instanceof Xe&&(n=n.right);const r=this.exec.getVariableName(n,t),i=t.getVariable(r);let o=e.args[1];const a=this.exec.evalExpression(o,t),l=i.value.getSubData(this.exec,n.postfix,t),u=new B(l.value,l.typeInfo);return l instanceof B&&a instanceof B&&(l.value-=a.value),i.value instanceof qe&&i.value.setDataValue(this.exec,l,n.postfix,t),u}AtomicMax(e,t){let n=e.args[0];n instanceof Xe&&(n=n.right);const r=this.exec.getVariableName(n,t),i=t.getVariable(r);let o=e.args[1];const a=this.exec.evalExpression(o,t),l=i.value.getSubData(this.exec,n.postfix,t),u=new B(l.value,l.typeInfo);return l instanceof B&&a instanceof B&&(l.value=Math.max(l.value,a.value)),i.value instanceof qe&&i.value.setDataValue(this.exec,l,n.postfix,t),u}AtomicMin(e,t){let n=e.args[0];n instanceof Xe&&(n=n.right);const r=this.exec.getVariableName(n,t),i=t.getVariable(r);let o=e.args[1];const a=this.exec.evalExpression(o,t),l=i.value.getSubData(this.exec,n.postfix,t),u=new B(l.value,l.typeInfo);return l instanceof B&&a instanceof B&&(l.value=Math.min(l.value,a.value)),i.value instanceof qe&&i.value.setDataValue(this.exec,l,n.postfix,t),u}AtomicAnd(e,t){let n=e.args[0];n instanceof Xe&&(n=n.right);const r=this.exec.getVariableName(n,t),i=t.getVariable(r);let o=e.args[1];const a=this.exec.evalExpression(o,t),l=i.value.getSubData(this.exec,n.postfix,t),u=new B(l.value,l.typeInfo);return l instanceof B&&a instanceof B&&(l.value=l.value&a.value),i.value instanceof qe&&i.value.setDataValue(this.exec,l,n.postfix,t),u}AtomicOr(e,t){let n=e.args[0];n instanceof Xe&&(n=n.right);const r=this.exec.getVariableName(n,t),i=t.getVariable(r);let o=e.args[1];const a=this.exec.evalExpression(o,t),l=i.value.getSubData(this.exec,n.postfix,t),u=new B(l.value,l.typeInfo);return l instanceof B&&a instanceof B&&(l.value=l.value|a.value),i.value instanceof qe&&i.value.setDataValue(this.exec,l,n.postfix,t),u}AtomicXor(e,t){let n=e.args[0];n instanceof Xe&&(n=n.right);const r=this.exec.getVariableName(n,t),i=t.getVariable(r);let o=e.args[1];const a=this.exec.evalExpression(o,t),l=i.value.getSubData(this.exec,n.postfix,t),u=new B(l.value,l.typeInfo);return l instanceof B&&a instanceof B&&(l.value=l.value^a.value),i.value instanceof qe&&i.value.setDataValue(this.exec,l,n.postfix,t),u}AtomicExchange(e,t){let n=e.args[0];n instanceof Xe&&(n=n.right);const r=this.exec.getVariableName(n,t),i=t.getVariable(r);let o=e.args[1];const a=this.exec.evalExpression(o,t),l=i.value.getSubData(this.exec,n.postfix,t),u=new B(l.value,l.typeInfo);return l instanceof B&&a instanceof B&&(l.value=a.value),i.value instanceof qe&&i.value.setDataValue(this.exec,l,n.postfix,t),u}AtomicCompareExchangeWeak(e,t){return console.error("TODO: atomicCompareExchangeWeak"),null}Pack4x8snorm(e,t){return console.error("TODO: pack4x8snorm"),null}Pack4x8unorm(e,t){return console.error("TODO: pack4x8unorm"),null}Pack4xI8(e,t){return console.error("TODO: pack4xI8"),null}Pack4xU8(e,t){return console.error("TODO: pack4xU8"),null}Pack4x8Clamp(e,t){return console.error("TODO: pack4x8Clamp"),null}Pack4xU8Clamp(e,t){return console.error("TODO: pack4xU8Clamp"),null}Pack2x16snorm(e,t){return console.error("TODO: pack2x16snorm"),null}Pack2x16unorm(e,t){return console.error("TODO: pack2x16unorm"),null}Pack2x16float(e,t){return console.error("TODO: pack2x16float"),null}Unpack4x8snorm(e,t){return console.error("TODO: unpack4x8snorm"),null}Unpack4x8unorm(e,t){return console.error("TODO: unpack4x8unorm"),null}Unpack4xI8(e,t){return console.error("TODO: unpack4xI8"),null}Unpack4xU8(e,t){return console.error("TODO: unpack4xU8"),null}Unpack2x16snorm(e,t){return console.error("TODO: unpack2x16snorm"),null}Unpack2x16unorm(e,t){return console.error("TODO: unpack2x16unorm"),null}Unpack2x16float(e,t){return console.error("TODO: unpack2x16float"),null}StorageBarrier(e,t){return null}TextureBarrier(e,t){return null}WorkgroupBarrier(e,t){return null}WorkgroupUniformLoad(e,t){return null}SubgroupAdd(e,t){return console.error("TODO: subgroupAdd"),null}SubgroupExclusiveAdd(e,t){return console.error("TODO: subgroupExclusiveAdd"),null}SubgroupInclusiveAdd(e,t){return console.error("TODO: subgroupInclusiveAdd"),null}SubgroupAll(e,t){return console.error("TODO: subgroupAll"),null}SubgroupAnd(e,t){return console.error("TODO: subgroupAnd"),null}SubgroupAny(e,t){return console.error("TODO: subgroupAny"),null}SubgroupBallot(e,t){return console.error("TODO: subgroupBallot"),null}SubgroupBroadcast(e,t){return console.error("TODO: subgroupBroadcast"),null}SubgroupBroadcastFirst(e,t){return console.error("TODO: subgroupBroadcastFirst"),null}SubgroupElect(e,t){return console.error("TODO: subgroupElect"),null}SubgroupMax(e,t){return console.error("TODO: subgroupMax"),null}SubgroupMin(e,t){return console.error("TODO: subgroupMin"),null}SubgroupMul(e,t){return console.error("TODO: subgroupMul"),null}SubgroupExclusiveMul(e,t){return console.error("TODO: subgroupExclusiveMul"),null}SubgroupInclusiveMul(e,t){return console.error("TODO: subgroupInclusiveMul"),null}SubgroupOr(e,t){return console.error("TODO: subgroupOr"),null}SubgroupShuffle(e,t){return console.error("TODO: subgroupShuffle"),null}SubgroupShuffleDown(e,t){return console.error("TODO: subgroupShuffleDown"),null}SubgroupShuffleUp(e,t){return console.error("TODO: subgroupShuffleUp"),null}SubgroupShuffleXor(e,t){return console.error("TODO: subgroupShuffleXor"),null}SubgroupXor(e,t){return console.error("TODO: subgroupXor"),null}QuadBroadcast(e,t){return console.error("TODO: quadBroadcast"),null}QuadSwapDiagonal(e,t){return console.error("TODO: quadSwapDiagonal"),null}QuadSwapX(e,t){return console.error("TODO: quadSwapX"),null}QuadSwapY(e,t){return console.error("TODO: quadSwapY"),null}}const bu={vec2:2,vec2f:2,vec2i:2,vec2u:2,vec2b:2,vec2h:2,vec3:3,vec3f:3,vec3i:3,vec3u:3,vec3b:3,vec3h:3,vec4:4,vec4f:4,vec4i:4,vec4u:4,vec4b:4,vec4h:4},It={mat2x2:[2,2,4],mat2x2f:[2,2,4],mat2x2h:[2,2,4],mat2x3:[2,3,6],mat2x3f:[2,3,6],mat2x3h:[2,3,6],mat2x4:[2,4,8],mat2x4f:[2,4,8],mat2x4h:[2,4,8],mat3x2:[3,2,6],mat3x2f:[3,2,6],mat3x2h:[3,2,6],mat3x3:[3,3,9],mat3x3f:[3,3,9],mat3x3h:[3,3,9],mat3x4:[3,4,12],mat3x4f:[3,4,12],mat3x4h:[3,4,12],mat4x2:[4,2,8],mat4x2f:[4,2,8],mat4x2h:[4,2,8],mat4x3:[4,3,12],mat4x3f:[4,3,12],mat4x3h:[4,3,12],mat4x4:[4,4,16],mat4x4f:[4,4,16],mat4x4h:[4,4,16]};class _t extends qA{constructor(e,t){var n;super(),this.ast=e??[],this.reflection=new Zn,this.reflection.updateAST(this.ast),this.context=(n=t?.clone())!==null&&n!==void 0?n:new Eh,this.builtins=new HA(this),this.typeInfo={bool:this.getTypeInfo(Y.bool),i32:this.getTypeInfo(Y.i32),u32:this.getTypeInfo(Y.u32),f32:this.getTypeInfo(Y.f32),f16:this.getTypeInfo(Y.f16),vec2f:this.getTypeInfo(z.vec2f),vec2u:this.getTypeInfo(z.vec2u),vec2i:this.getTypeInfo(z.vec2i),vec2h:this.getTypeInfo(z.vec2h),vec3f:this.getTypeInfo(z.vec3f),vec3u:this.getTypeInfo(z.vec3u),vec3i:this.getTypeInfo(z.vec3i),vec3h:this.getTypeInfo(z.vec3h),vec4f:this.getTypeInfo(z.vec4f),vec4u:this.getTypeInfo(z.vec4u),vec4i:this.getTypeInfo(z.vec4i),vec4h:this.getTypeInfo(z.vec4h),mat2x2f:this.getTypeInfo(z.mat2x2f),mat2x3f:this.getTypeInfo(z.mat2x3f),mat2x4f:this.getTypeInfo(z.mat2x4f),mat3x2f:this.getTypeInfo(z.mat3x2f),mat3x3f:this.getTypeInfo(z.mat3x3f),mat3x4f:this.getTypeInfo(z.mat3x4f),mat4x2f:this.getTypeInfo(z.mat4x2f),mat4x3f:this.getTypeInfo(z.mat4x3f),mat4x4f:this.getTypeInfo(z.mat4x4f)}}getVariableValue(e){var t,n;const r=(n=(t=this.context.getVariable(e))===null||t===void 0?void 0:t.value)!==null&&n!==void 0?n:null;if(r===null)return null;if(r instanceof B)return r.value;if(r instanceof P||r instanceof de)return Array.from(r.data);if(r instanceof qe&&r.typeInfo instanceof Ps){if(r.typeInfo.format.name==="u32")return Array.from(new Uint32Array(r.buffer,r.offset,r.typeInfo.count));if(r.typeInfo.format.name==="i32")return Array.from(new Int32Array(r.buffer,r.offset,r.typeInfo.count));if(r.typeInfo.format.name==="f32")return Array.from(new Float32Array(r.buffer,r.offset,r.typeInfo.count))}return console.error("Unsupported return variable type "+r.typeInfo.name),null}execute(e){(e=e??{}).constants&&this._setOverrides(e.constants,this.context),this._execStatements(this.ast,this.context)}dispatchWorkgroups(e,t,n,r){const i=this.context.clone();(r=r??{}).constants&&this._setOverrides(r.constants,i),this._execStatements(this.ast,i);const o=i.getFunction(e);if(!o)return void console.error(`Function ${e} not found`);if(typeof t=="number")t=[t,1,1];else{if(t.length===0)return void console.error("Invalid dispatch count");t.length===1?t=[t[0],1,1]:t.length===2?t=[t[0],t[1],1]:t.length>3&&(t=[t[0],t[1],t[2]])}const a=t[0],l=t[1],u=t[2],c=this.getTypeInfo("vec3u");i.setVariable("@num_workgroups",new P(t,c));for(const h in n)for(const d in n[h]){const w=n[h][d];i.variables.forEach(k=>{var A;const m=k.node;if(m!=null&&m.attributes){let S=null,b=null;for(const f of m.attributes)f.name==="binding"?S=f.value:f.name==="group"&&(b=f.value);if(d==S&&h==b)if(w.texture!==void 0&&w.descriptor!==void 0){const f=new hs(w.texture,this.getTypeInfo(m.type),w.descriptor,(A=w.texture.view)!==null&&A!==void 0?A:null);k.value=f}else w.uniform!==void 0?k.value=new qe(w.uniform,this.getTypeInfo(m.type)):k.value=new qe(w,this.getTypeInfo(m.type))}})}for(let h=0;h<u;++h)for(let d=0;d<l;++d)for(let w=0;w<a;++w)i.setVariable("@workgroup_id",new P([w,d,h],this.getTypeInfo("vec3u"))),this._dispatchWorkgroup(o,[w,d,h],i)}execStatement(e,t){if(e instanceof Ty)return this.evalExpression(e.value,t);if(e instanceof Cy){if(e.condition){const n=this.evalExpression(e.condition,t);if(!(n instanceof B))throw Error("Invalid break-if condition");if(!n.value)return null}return _t._breakObj}if(e instanceof Ny)return _t._continueObj;if(e instanceof oo)this._let(e,t);else if(e instanceof ms)this._var(e,t);else if(e instanceof Ea)this._const(e,t);else if(e instanceof Eo)this._function(e,t);else{if(e instanceof Ey)return this._if(e,t);if(e instanceof Iy)return this._switch(e,t);if(e instanceof _y)return this._for(e,t);if(e instanceof xy)return this._while(e,t);if(e instanceof ky)return this._loop(e,t);if(e instanceof nc){const n=t.clone();return n.currentFunctionName=t.currentFunctionName,this._execStatements(e.body,n)}if(e instanceof Sy)this._assign(e,t);else if(e instanceof vy)this._increment(e,t);else{if(e instanceof cs)return null;if(e instanceof wh){const n=e.name;t.getVariable(n)===null&&t.setVariable(n,new B(0,this.getTypeInfo("u32")))}else if(e instanceof xh)this._call(e,t);else{if(e instanceof Ay||e instanceof _h)return null;console.error("Invalid statement type.",e,"Line "+e.line)}}}return null}evalExpression(e,t){return e instanceof bn?this._evalBinaryOp(e,t):e instanceof et?this._evalLiteral(e,t):e instanceof Yt?this._evalVariable(e,t):e instanceof vh?this._evalCall(e,t):e instanceof Hn?this._evalCreate(e,t):e instanceof $y?this._evalConst(e,t):e instanceof Dy?this._evalBitcast(e,t):e instanceof Xe?this._evalUnaryOp(e,t):(console.error("Invalid expression type",e,"Line "+e.line),null)}getTypeInfo(e){var t;if(e instanceof Y){const r=this.reflection.getTypeInfo(e);if(r!==null)return r}let n=(t=this.typeInfo[e])!==null&&t!==void 0?t:null;return n!==null||(n=this.reflection.getTypeInfoByName(e)),n}_setOverrides(e,t){for(const n in e){const r=e[n],i=this.reflection.getOverrideInfo(n);i!==null?(i.type===null&&(i.type=this.getTypeInfo("u32")),i.type.name==="u32"||i.type.name==="i32"||i.type.name==="f32"||i.type.name==="f16"?t.setVariable(n,new B(r,i.type)):i.type.name==="bool"?t.setVariable(n,new B(r?1:0,i.type)):i.type.name==="vec2"||i.type.name==="vec3"||i.type.name==="vec4"||i.type.name==="vec2f"||i.type.name==="vec3f"||i.type.name==="vec4f"||i.type.name==="vec2i"||i.type.name==="vec3i"||i.type.name==="vec4i"||i.type.name==="vec2u"||i.type.name==="vec3u"||i.type.name==="vec4u"||i.type.name==="vec2h"||i.type.name==="vec3h"||i.type.name==="vec4h"?t.setVariable(n,new P(r,i.type)):console.error("Invalid constant type for "+n)):console.error(`Override ${n} does not exist in the shader.`)}}_dispatchWorkgroup(e,t,n){const r=[1,1,1];for(const c of e.node.attributes)if(c.name==="workgroup_size"){if(c.value.length>0){const h=n.getVariableValue(c.value[0]);r[0]=h instanceof B?h.value:parseInt(c.value[0])}if(c.value.length>1){const h=n.getVariableValue(c.value[1]);r[1]=h instanceof B?h.value:parseInt(c.value[1])}if(c.value.length>2){const h=n.getVariableValue(c.value[2]);r[2]=h instanceof B?h.value:parseInt(c.value[2])}}const i=this.getTypeInfo("vec3u"),o=this.getTypeInfo("u32");n.setVariable("@workgroup_size",new P(r,i));const a=r[0],l=r[1],u=r[2];for(let c=0,h=0;c<u;++c)for(let d=0;d<l;++d)for(let w=0;w<a;++w,++h){const k=[w,d,c],A=[w+t[0]*r[0],d+t[1]*r[1],c+t[2]*r[2]];n.setVariable("@local_invocation_id",new P(k,i)),n.setVariable("@global_invocation_id",new P(A,i)),n.setVariable("@local_invocation_index",new B(h,o)),this._dispatchExec(e,n)}}_dispatchExec(e,t){for(const n of e.node.args)for(const r of n.attributes)if(r.name==="builtin"){const i="@"+r.value,o=t.getVariable(i);o!==void 0&&t.variables.set(n.name,o)}this._execStatements(e.node.body,t)}getVariableName(e,t){for(;e instanceof Xe;)e=e.right;return e instanceof Yt?e.name:(console.error("Unknown variable type",e,"Line",e.line),null)}_execStatements(e,t){for(const n of e){if(n instanceof Array){const i=t.clone(),o=this._execStatements(n,i);if(o)return o;continue}const r=this.execStatement(n,t);if(r)return r}return null}_call(e,t){const n=t.clone();n.currentFunctionName=e.name;const r=t.getFunction(e.name);if(r){for(let i=0;i<r.node.args.length;++i){const o=r.node.args[i],a=this.evalExpression(e.args[i],n);n.setVariable(o.name,a,o)}this._execStatements(r.node.body,n)}else e.isBuiltin?this._callBuiltinFunction(e,n):this.getTypeInfo(e.name)&&this._evalCreate(e,t)}_increment(e,t){const n=this.getVariableName(e.variable,t),r=t.getVariable(n);r?e.operator==="++"?r.value instanceof B?r.value.value++:console.error(`Variable ${n} is not a scalar. Line ${e.line}`):e.operator==="--"?r.value instanceof B?r.value.value--:console.error(`Variable ${n} is not a scalar. Line ${e.line}`):console.error(`Unknown increment operator ${e.operator}. Line ${e.line}`):console.error(`Variable ${n} not found. Line ${e.line}`)}_getVariableData(e,t){if(e instanceof Yt){const n=this.getVariableName(e,t),r=t.getVariable(n);return r===null?(console.error(`Variable ${n} not found. Line ${e.line}`),null):r.value.getSubData(this,e.postfix,t)}if(e instanceof Xe){if(e.operator==="*"){const n=this._getVariableData(e.right,t);return n instanceof Mr?n.reference.getSubData(this,e.postfix,t):(console.error(`Variable ${e.right} is not a pointer. Line ${e.line}`),null)}if(e.operator==="&"){const n=this._getVariableData(e.right,t);return new Mr(n)}}return null}_assign(e,t){let n=null,r="<var>",i=null;if(e.variable instanceof Xe){const l=this._getVariableData(e.variable,t),u=this.evalExpression(e.value,t),c=e.operator;if(c==="="){if(l instanceof B||l instanceof P||l instanceof de){if(u instanceof B||u instanceof P||u instanceof de&&l.data.length===u.data.length)return void l.data.set(u.data);console.error("Invalid assignment. Line "+e.line)}else if(l instanceof qe&&u instanceof qe&&l.buffer.byteLength-l.offset>=u.buffer.byteLength-u.offset)return void(l.buffer.byteLength%4==0?new Uint32Array(l.buffer,l.offset,l.typeInfo.size/4).set(new Uint32Array(u.buffer,u.offset,u.typeInfo.size/4)):new Uint8Array(l.buffer,l.offset,l.typeInfo.size).set(new Uint8Array(u.buffer,u.offset,u.typeInfo.size)));return console.error("Invalid assignment. Line "+e.line),null}if(c==="+=")return l instanceof B||l instanceof P||l instanceof de?u instanceof B||u instanceof P||u instanceof de?void l.data.set(u.data.map((h,d)=>l.data[d]+h)):void console.error("Invalid assignment . Line "+e.line):void console.error("Invalid assignment. Line "+e.line);if(c==="-=")return(l instanceof B||l instanceof P||l instanceof de)&&(u instanceof B||u instanceof P||u instanceof de)?void l.data.set(u.data.map((h,d)=>l.data[d]-h)):void console.error("Invalid assignment. Line "+e.line)}if(e.variable instanceof Xe){if(e.variable.operator==="*"){r=this.getVariableName(e.variable.right,t);const l=t.getVariable(r);if(!(l&&l.value instanceof Mr))return void console.error(`Variable ${r} is not a pointer. Line ${e.line}`);n=l.value.reference;let u=e.variable.postfix;if(!u){let c=e.variable.right;for(;c instanceof Xe;){if(c.postfix){u=c.postfix;break}c=c.right}}u&&(n=n.getSubData(this,u,t))}}else{i=e.variable.postfix,r=this.getVariableName(e.variable,t);const l=t.getVariable(r);if(l===null)return void console.error(`Variable ${r} not found. Line ${e.line}`);n=l.value}if(n instanceof Mr&&(n=n.reference),n===null)return void console.error(`Variable ${r} not found. Line ${e.line}`);const o=this.evalExpression(e.value,t),a=e.operator;if(a==="=")if(n instanceof qe)n.setDataValue(this,o,i,t);else if(i){if(!(n instanceof P||n instanceof de))return void console.error(`Variable ${r} is not a vector or matrix. Line ${e.line}`);if(i instanceof gi){const l=this.evalExpression(i.index,t).value;if(n instanceof P){if(!(o instanceof B))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[l]=o.value}else{if(!(n instanceof de))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);{const u=this.evalExpression(i.index,t).value;if(u<0)return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);if(!(o instanceof P))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);{const c=n.typeInfo.getTypeName();if(c==="mat2x2"||c==="mat2x2f"||c==="mat2x2h"){if(!(u<2&&o.data.length===2))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[2*u]=o.data[0],n.data[2*u+1]=o.data[1]}else if(c==="mat2x3"||c==="mat2x3f"||c==="mat2x3h"){if(!(u<2&&o.data.length===3))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[3*u]=o.data[0],n.data[3*u+1]=o.data[1],n.data[3*u+2]=o.data[2]}else if(c==="mat2x4"||c==="mat2x4f"||c==="mat2x4h"){if(!(u<2&&o.data.length===4))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[4*u]=o.data[0],n.data[4*u+1]=o.data[1],n.data[4*u+2]=o.data[2],n.data[4*u+3]=o.data[3]}else if(c==="mat3x2"||c==="mat3x2f"||c==="mat3x2h"){if(!(u<3&&o.data.length===2))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[2*u]=o.data[0],n.data[2*u+1]=o.data[1]}else if(c==="mat3x3"||c==="mat3x3f"||c==="mat3x3h"){if(!(u<3&&o.data.length===3))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[3*u]=o.data[0],n.data[3*u+1]=o.data[1],n.data[3*u+2]=o.data[2]}else if(c==="mat3x4"||c==="mat3x4f"||c==="mat3x4h"){if(!(u<3&&o.data.length===4))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[4*u]=o.data[0],n.data[4*u+1]=o.data[1],n.data[4*u+2]=o.data[2],n.data[4*u+3]=o.data[3]}else if(c==="mat4x2"||c==="mat4x2f"||c==="mat4x2h"){if(!(u<4&&o.data.length===2))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[2*u]=o.data[0],n.data[2*u+1]=o.data[1]}else if(c==="mat4x3"||c==="mat4x3f"||c==="mat4x3h"){if(!(u<4&&o.data.length===3))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[3*u]=o.data[0],n.data[3*u+1]=o.data[1],n.data[3*u+2]=o.data[2]}else{if(c!=="mat4x4"&&c!=="mat4x4f"&&c!=="mat4x4h")return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);if(!(u<4&&o.data.length===4))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[4*u]=o.data[0],n.data[4*u+1]=o.data[1],n.data[4*u+2]=o.data[2],n.data[4*u+3]=o.data[3]}}}}}else if(i instanceof Cr){const l=i.value;if(!(n instanceof P))return void console.error(`Invalid assignment to ${l}. Variable ${r} is not a vector. Line ${e.line}`);if(o instanceof B){if(l.length>1)return void console.error(`Invalid assignment to ${l} for variable ${r}. Line ${e.line}`);if(l==="x")n.data[0]=o.value;else if(l==="y"){if(n.data.length<2)return void console.error(`Invalid assignment to ${l} for variable ${r}. Line ${e.line}`);n.data[1]=o.value}else if(l==="z"){if(n.data.length<3)return void console.error(`Invalid assignment to ${l} for variable ${r}. Line ${e.line}`);n.data[2]=o.value}else if(l==="w"){if(n.data.length<4)return void console.error(`Invalid assignment to ${l} for variable ${r}. Line ${e.line}`);n.data[3]=o.value}}else{if(!(o instanceof P))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);if(l.length!==o.data.length)return void console.error(`Invalid assignment to ${l} for variable ${r}. Line ${e.line}`);for(let u=0;u<l.length;++u){const c=l[u];if(c==="x"||c==="r")n.data[0]=o.data[u];else if(c==="y"||c==="g"){if(o.data.length<2)return void console.error(`Invalid assignment to ${c} for variable ${r}. Line ${e.line}`);n.data[1]=o.data[u]}else if(c==="z"||c==="b"){if(o.data.length<3)return void console.error(`Invalid assignment to ${c} for variable ${r}. Line ${e.line}`);n.data[2]=o.data[u]}else{if(c!=="w"&&c!=="a")return void console.error(`Invalid assignment to ${c} for variable ${r}. Line ${e.line}`);if(o.data.length<4)return void console.error(`Invalid assignment to ${c} for variable ${r}. Line ${e.line}`);n.data[3]=o.data[u]}}}}}else n instanceof B&&o instanceof B?n.value=o.value:n instanceof P&&o instanceof P||n instanceof de&&o instanceof de?n.data.set(o.data):console.error(`Invalid assignment to ${r}. Line ${e.line}`);else{const l=n.getSubData(this,i,t);if(l instanceof P&&o instanceof B){const u=l.data,c=o.value;if(a==="+=")for(let h=0;h<u.length;++h)u[h]+=c;else if(a==="-=")for(let h=0;h<u.length;++h)u[h]-=c;else if(a==="*=")for(let h=0;h<u.length;++h)u[h]*=c;else if(a==="/=")for(let h=0;h<u.length;++h)u[h]/=c;else if(a==="%=")for(let h=0;h<u.length;++h)u[h]%=c;else if(a==="&=")for(let h=0;h<u.length;++h)u[h]&=c;else if(a==="|=")for(let h=0;h<u.length;++h)u[h]|=c;else if(a==="^=")for(let h=0;h<u.length;++h)u[h]^=c;else if(a==="<<=")for(let h=0;h<u.length;++h)u[h]<<=c;else if(a===">>=")for(let h=0;h<u.length;++h)u[h]>>=c;else console.error(`Invalid operator ${a}. Line ${e.line}`)}else if(l instanceof P&&o instanceof P){const u=l.data,c=o.data;if(u.length!==c.length)return void console.error("Vector length mismatch. Line "+e.line);if(a==="+=")for(let h=0;h<u.length;++h)u[h]+=c[h];else if(a==="-=")for(let h=0;h<u.length;++h)u[h]-=c[h];else if(a==="*=")for(let h=0;h<u.length;++h)u[h]*=c[h];else if(a==="/=")for(let h=0;h<u.length;++h)u[h]/=c[h];else if(a==="%=")for(let h=0;h<u.length;++h)u[h]%=c[h];else if(a==="&=")for(let h=0;h<u.length;++h)u[h]&=c[h];else if(a==="|=")for(let h=0;h<u.length;++h)u[h]|=c[h];else if(a==="^=")for(let h=0;h<u.length;++h)u[h]^=c[h];else if(a==="<<=")for(let h=0;h<u.length;++h)u[h]<<=c[h];else if(a===">>=")for(let h=0;h<u.length;++h)u[h]>>=c[h];else console.error(`Invalid operator ${a}. Line ${e.line}`)}else{if(!(l instanceof B&&o instanceof B))return void console.error(`Invalid type for ${e.operator} operator. Line ${e.line}`);a==="+="?l.value+=o.value:a==="-="?l.value-=o.value:a==="*="?l.value*=o.value:a==="/="?l.value/=o.value:a==="%="?l.value%=o.value:a==="&="?l.value&=o.value:a==="|="?l.value|=o.value:a==="^="?l.value^=o.value:a==="<<="?l.value<<=o.value:a===">>="?l.value>>=o.value:console.error(`Invalid operator ${a}. Line ${e.line}`)}n instanceof qe&&n.setDataValue(this,l,i,t)}}_function(e,t){const n=new Ih(e);t.functions.set(e.name,n)}_const(e,t){let n=null;e.value!==null&&(n=this.evalExpression(e.value,t)),t.createVariable(e.name,n,e)}_let(e,t){let n=null;if(e.value!==null){if(n=this.evalExpression(e.value,t),n===null)return void console.error(`Invalid value for variable ${e.name}. Line ${e.line}`);e.value instanceof Xe||(n=n.clone())}else{const r=e.type.name;if(r==="f32"||r==="i32"||r==="u32"||r==="bool"||r==="f16"||r==="vec2"||r==="vec3"||r==="vec4"||r==="vec2f"||r==="vec3f"||r==="vec4f"||r==="vec2i"||r==="vec3i"||r==="vec4i"||r==="vec2u"||r==="vec3u"||r==="vec4u"||r==="vec2h"||r==="vec3h"||r==="vec4h"||r==="vec2b"||r==="vec3b"||r==="vec4b"||r==="mat2x2"||r==="mat2x3"||r==="mat2x4"||r==="mat3x2"||r==="mat3x3"||r==="mat3x4"||r==="mat4x2"||r==="mat4x3"||r==="mat4x4"||r==="mat2x2f"||r==="mat2x3f"||r==="mat2x4f"||r==="mat3x2f"||r==="mat3x3f"||r==="mat3x4f"||r==="mat4x2f"||r==="mat4x3f"||r==="mat4x4f"||r==="mat2x2h"||r==="mat2x3h"||r==="mat2x4h"||r==="mat3x2h"||r==="mat3x3h"||r==="mat3x4h"||r==="mat4x2h"||r==="mat4x3h"||r==="mat4x4h"||r==="array"){const i=new Hn(e.type,[]);n=this._evalCreate(i,t)}}t.createVariable(e.name,n,e)}_var(e,t){let n=null;if(e.value!==null){if(n=this.evalExpression(e.value,t),n===null)return void console.error(`Invalid value for variable ${e.name}. Line ${e.line}`);e.value instanceof Xe||(n=n.clone())}else{if(e.type===null)return void console.error(`Variable ${e.name} has no type. Line ${e.line}`);const r=e.type.name;if(r==="f32"||r==="i32"||r==="u32"||r==="bool"||r==="f16"||r==="vec2"||r==="vec3"||r==="vec4"||r==="vec2f"||r==="vec3f"||r==="vec4f"||r==="vec2i"||r==="vec3i"||r==="vec4i"||r==="vec2u"||r==="vec3u"||r==="vec4u"||r==="vec2h"||r==="vec3h"||r==="vec4h"||r==="vec2b"||r==="vec3b"||r==="vec4b"||r==="mat2x2"||r==="mat2x3"||r==="mat2x4"||r==="mat3x2"||r==="mat3x3"||r==="mat3x4"||r==="mat4x2"||r==="mat4x3"||r==="mat4x4"||r==="mat2x2f"||r==="mat2x3f"||r==="mat2x4f"||r==="mat3x2f"||r==="mat3x3f"||r==="mat3x4f"||r==="mat4x2f"||r==="mat4x3f"||r==="mat4x4f"||r==="mat2x2h"||r==="mat2x3h"||r==="mat2x4h"||r==="mat3x2h"||r==="mat3x3h"||r==="mat3x4h"||r==="mat4x2h"||r==="mat4x3h"||r==="mat4x4h"||e.type instanceof ao||e.type instanceof cs||e.type instanceof z){const i=new Hn(e.type,[]);n=this._evalCreate(i,t)}}t.createVariable(e.name,n,e)}_switch(e,t){t=t.clone();const n=this.evalExpression(e.condition,t);if(!(n instanceof B))return console.error("Invalid if condition. Line "+e.line),null;let r=null;for(const i of e.cases)if(i instanceof Py)for(const o of i.selectors){if(o instanceof Aa){r=i;continue}const a=this.evalExpression(o,t);if(!(a instanceof B))return console.error("Invalid case selector. Line "+e.line),null;if(a.value===n.value)return this._execStatements(i.body,t)}else i instanceof Ry&&(r=i);return r?this._execStatements(r.body,t):null}_if(e,t){t=t.clone();const n=this.evalExpression(e.condition,t);if(!(n instanceof B))return console.error("Invalid if condition. Line "+e.line),null;if(n.value)return this._execStatements(e.body,t);for(const r of e.elseif){const i=this.evalExpression(r.condition,t);if(!(i instanceof B))return console.error("Invalid if condition. Line "+e.line),null;if(i.value)return this._execStatements(r.body,t)}return e.else?this._execStatements(e.else,t):null}_getScalarValue(e){return e instanceof B?e.value:(console.error("Expected scalar value.",e),0)}_for(e,t){for(t=t.clone(),this.execStatement(e.init,t);this._getScalarValue(this.evalExpression(e.condition,t));){const n=this._execStatements(e.body,t);if(n===_t._breakObj)break;if(n!==null&&n!==_t._continueObj)return n;this.execStatement(e.increment,t)}return null}_loop(e,t){for(t=t.clone();;){const n=this._execStatements(e.body,t);if(n===_t._breakObj)break;if(n===_t._continueObj){if(e.continuing&&this._execStatements(e.continuing.body,t)===_t._breakObj)break}else if(n!==null)return n}return null}_while(e,t){for(t=t.clone();this._getScalarValue(this.evalExpression(e.condition,t));){const n=this._execStatements(e.body,t);if(n===_t._breakObj)break;if(n!==_t._continueObj&&n!==null)return n}return null}_evalBitcast(e,t){const n=this.evalExpression(e.value,t),r=e.type;if(n instanceof B){const i=Td(n.value,n.typeInfo.name,r.name);return new B(i,this.getTypeInfo(r))}if(n instanceof P){const i=n.typeInfo.getTypeName();let o="";if(i.endsWith("f"))o="f32";else if(i.endsWith("i"))o="i32";else if(i.endsWith("u"))o="u32";else if(i.endsWith("b"))o="bool";else{if(!i.endsWith("h"))return console.error(`Unknown vector type ${i}. Line ${e.line}`),null;o="f16"}const a=r.getTypeName();let l="";if(a.endsWith("f"))l="f32";else if(a.endsWith("i"))l="i32";else if(a.endsWith("u"))l="u32";else if(a.endsWith("b"))l="bool";else{if(!a.endsWith("h"))return console.error(`Unknown vector type ${l}. Line ${e.line}`),null;l="f16"}const u=((c,h,d)=>{if(h===d)return c;const w=Array(c.length);for(let k=0;k<c.length;k++)w[k]=Td(c[k],h,d);return w})(Array.from(n.data),o,l);return new P(u,this.getTypeInfo(r))}return console.error(`TODO: bitcast for ${n.typeInfo.name}. Line ${e.line}`),null}_evalConst(e,t){return t.getVariableValue(e.name).clone().getSubData(this,e.postfix,t)}_evalCreate(e,t){var n;if(e instanceof Hn){if(e.type===null)return sc.void;switch(e.type.getTypeName()){case"bool":case"i32":case"u32":case"f32":case"f16":return this._callConstructorValue(e,t);case"vec2":case"vec3":case"vec4":case"vec2f":case"vec3f":case"vec4f":case"vec2h":case"vec3h":case"vec4h":case"vec2i":case"vec3i":case"vec4i":case"vec2u":case"vec3u":case"vec4u":case"vec2b":case"vec3b":case"vec4b":return this._callConstructorVec(e,t);case"mat2x2":case"mat2x2f":case"mat2x2h":case"mat2x3":case"mat2x3f":case"mat2x3h":case"mat2x4":case"mat2x4f":case"mat2x4h":case"mat3x2":case"mat3x2f":case"mat3x2h":case"mat3x3":case"mat3x3f":case"mat3x3h":case"mat3x4":case"mat3x4f":case"mat3x4h":case"mat4x2":case"mat4x2f":case"mat4x2h":case"mat4x3":case"mat4x3f":case"mat4x3h":case"mat4x4":case"mat4x4f":case"mat4x4h":return this._callConstructorMatrix(e,t)}}const r=e instanceof Hn?e.type.name:e.name,i=e instanceof Hn?this.getTypeInfo(e.type):this.getTypeInfo(e.name);if(i===null)return console.error(`Unknown type ${r}. Line ${e.line}`),null;if(i.size===0)return null;const o=new qe(new ArrayBuffer(i.size),i,0);if(i instanceof Ns){if(e.args)for(let a=0;a<e.args.length;++a){const l=i.members[a],u=e.args[a],c=this.evalExpression(u,t);o.setData(this,c,l.type,l.offset,t)}}else if(i instanceof Ps){let a=0;if(e.args)for(let l=0;l<e.args.length;++l){const u=e.args[l],c=this.evalExpression(u,t);i.format===null&&(((n=c.typeInfo)===null||n===void 0?void 0:n.name)==="x32"?i.format=this.getTypeInfo("i32"):i.format=c.typeInfo),o.setData(this,c,i.format,a,t),a+=i.stride}}else console.error(`Unknown type "${r}". Line ${e.line}`);return e instanceof Hn?o.getSubData(this,e.postfix,t):o}_evalLiteral(e,t){const n=this.getTypeInfo(e.type),r=n.name;return r==="x32"||r==="u32"||r==="f32"||r==="f16"||r==="i32"||r==="bool"?new B(e.scalarValue,n):r==="vec2"||r==="vec3"||r==="vec4"||r==="vec2f"||r==="vec3f"||r==="vec4f"||r==="vec2h"||r==="vec3h"||r==="vec4h"||r==="vec2i"||r==="vec3i"||r==="vec4i"||r==="vec2u"||r==="vec3u"||r==="vec4u"?this._callConstructorVec(e,t):r==="mat2x2"||r==="mat2x3"||r==="mat2x4"||r==="mat3x2"||r==="mat3x3"||r==="mat3x4"||r==="mat4x2"||r==="mat4x3"||r==="mat4x4"||r==="mat2x2f"||r==="mat2x3f"||r==="mat2x4f"||r==="mat3x2f"||r==="mat3x3f"||r==="mat3x4f"||r==="mat4x2f"||r==="mat4x3f"||r==="mat4x4f"||r==="mat2x2h"||r==="mat2x3h"||r==="mat2x4h"||r==="mat3x2h"||r==="mat3x3h"||r==="mat3x4h"||r==="mat4x2h"||r==="mat4x3h"||r==="mat4x4h"?this._callConstructorMatrix(e,t):e.value}_evalVariable(e,t){const n=t.getVariableValue(e.name);return n===null?n:n.getSubData(this,e.postfix,t)}_maxFormatTypeInfo(e){let t=e[0];if(t.name==="f32")return t;for(let n=1;n<e.length;++n){const r=_t._priority.get(t.name);_t._priority.get(e[n].name)<r&&(t=e[n])}return t.name==="x32"?this.getTypeInfo("i32"):t}_evalUnaryOp(e,t){const n=this.evalExpression(e.right,t);if(e.operator==="&")return new Mr(n);if(e.operator==="*")return n instanceof Mr?n.reference.getSubData(this,e.postfix,t):(console.error("Invalid dereference. Line "+e.line),null);const r=n instanceof B?n.value:n instanceof P?Array.from(n.data):null;switch(e.operator){case"+":{if(re(r)){const a=r.map((l,u)=>+l);return new P(a,n.typeInfo)}const i=r,o=this._maxFormatTypeInfo([n.typeInfo,n.typeInfo]);return new B(+i,o)}case"-":{if(re(r)){const a=r.map((l,u)=>-l);return new P(a,n.typeInfo)}const i=r,o=this._maxFormatTypeInfo([n.typeInfo,n.typeInfo]);return new B(-i,o)}case"!":{if(re(r)){const a=r.map((l,u)=>l?0:1);return new P(a,n.typeInfo)}const i=r,o=this._maxFormatTypeInfo([n.typeInfo,n.typeInfo]);return new B(i?0:1,o)}case"~":{if(re(r)){const a=r.map((l,u)=>~l);return new P(a,n.typeInfo)}const i=r,o=this._maxFormatTypeInfo([n.typeInfo,n.typeInfo]);return new B(~i,o)}}return console.error(`Invalid unary operator ${e.operator}. Line ${e.line}`),null}_evalBinaryOp(e,t){const n=this.evalExpression(e.left,t),r=this.evalExpression(e.right,t),i=n instanceof B?n.value:n instanceof P||n instanceof de?Array.from(n.data):null,o=r instanceof B?r.value:r instanceof P||r instanceof de?Array.from(r.data):null;switch(e.operator){case"+":{if(re(i)&&re(o)){const c=i,h=o;if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,k)=>w+h[k]);return new P(d,n.typeInfo)}if(re(i)){const c=o,h=i.map((d,w)=>d+c);return new P(h,n.typeInfo)}if(re(o)){const c=i,h=o.map((d,w)=>c+d);return new P(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new B(a+l,u)}case"-":{if(re(i)&&re(o)){const c=i,h=o;if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,k)=>w-h[k]);return new P(d,n.typeInfo)}if(re(i)){const c=o,h=i.map((d,w)=>d-c);return new P(h,n.typeInfo)}if(re(o)){const c=i,h=o.map((d,w)=>c-d);return new P(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new B(a-l,u)}case"*":{if(re(i)&&re(o)){const c=i,h=o;if(n instanceof de&&r instanceof de){const d=((m,S,b,f)=>{if(It[S.name]===void 0||It[f.name]===void 0)return null;const v=It[S.name][0],_=It[S.name][1],E=It[f.name][0];if(v!==It[f.name][1])return null;const D=Array(E*_);for(let M=0;M<_;M++)for(let $=0;$<E;$++){let C=0;for(let g=0;g<v;g++)C+=m[g*_+M]*b[$*v+g];D[M*E+$]=C}return D})(c,n.typeInfo,h,r.typeInfo);if(d===null)return console.error(`Matrix multiplication failed. Line ${e.line}.`),null;const w=It[r.typeInfo.name][0],k=It[n.typeInfo.name][1],A=this.getTypeInfo(`mat${w}x${k}f`);return new de(d,A)}if(n instanceof de&&r instanceof P){const d=((w,k,A,m)=>{if(It[k.name]===void 0||bu[m.name]===void 0)return null;const S=It[k.name][0],b=It[k.name][1];if(S!==A.length)return null;const f=Array(b);for(let v=0;v<b;v++){let _=0;for(let E=0;E<S;E++)_+=w[E*b+v]*A[E];f[v]=_}return f})(c,n.typeInfo,h,r.typeInfo);return d===null?(console.error(`Matrix vector multiplication failed. Line ${e.line}.`),null):new P(d,r.typeInfo)}if(n instanceof P&&r instanceof de){const d=((w,k,A,m)=>{if(bu[k.name]===void 0||It[m.name]===void 0)return null;const S=It[m.name][0],b=It[m.name][1];if(b!==w.length)return null;const f=[];for(let v=0;v<S;v++){let _=0;for(let E=0;E<b;E++)_+=w[E]*A[E*S+v];f[v]=_}return f})(c,n.typeInfo,h,r.typeInfo);return d===null?(console.error(`Matrix vector multiplication failed. Line ${e.line}.`),null):new P(d,n.typeInfo)}{if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,k)=>w*h[k]);return new P(d,n.typeInfo)}}if(re(i)){const c=o,h=i.map((d,w)=>d*c);return n instanceof de?new de(h,n.typeInfo):new P(h,n.typeInfo)}if(re(o)){const c=i,h=o.map((d,w)=>c*d);return r instanceof de?new de(h,r.typeInfo):new P(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new B(a*l,u)}case"%":{if(re(i)&&re(o)){const c=i,h=o;if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,k)=>w%h[k]);return new P(d,n.typeInfo)}if(re(i)){const c=o,h=i.map((d,w)=>d%c);return new P(h,n.typeInfo)}if(re(o)){const c=i,h=o.map((d,w)=>c%d);return new P(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new B(a%l,u)}case"/":{if(re(i)&&re(o)){const c=i,h=o;if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,k)=>w/h[k]);return new P(d,n.typeInfo)}if(re(i)){const c=o,h=i.map((d,w)=>d/c);return new P(h,n.typeInfo)}if(re(o)){const c=i,h=o.map((d,w)=>c/d);return new P(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new B(a/l,u)}case"&":{if(re(i)&&re(o)){const c=i,h=o;if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,k)=>w&h[k]);return new P(d,n.typeInfo)}if(re(i)){const c=o,h=i.map((d,w)=>d&c);return new P(h,n.typeInfo)}if(re(o)){const c=i,h=o.map((d,w)=>c&d);return new P(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new B(a&l,u)}case"|":{if(re(i)&&re(o)){const c=i,h=o;if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,k)=>w|h[k]);return new P(d,n.typeInfo)}if(re(i)){const c=o,h=i.map((d,w)=>d|c);return new P(h,n.typeInfo)}if(re(o)){const c=i,h=o.map((d,w)=>c|d);return new P(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new B(a|l,u)}case"^":{if(re(i)&&re(o)){const c=i,h=o;if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,k)=>w^h[k]);return new P(d,n.typeInfo)}if(re(i)){const c=o,h=i.map((d,w)=>d^c);return new P(h,n.typeInfo)}if(re(o)){const c=i,h=o.map((d,w)=>c^d);return new P(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new B(a^l,u)}case"<<":{if(re(i)&&re(o)){const c=i,h=o;if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,k)=>w<<h[k]);return new P(d,n.typeInfo)}if(re(i)){const c=o,h=i.map((d,w)=>d<<c);return new P(h,n.typeInfo)}if(re(o)){const c=i,h=o.map((d,w)=>c<<d);return new P(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new B(a<<l,u)}case">>":{if(re(i)&&re(o)){const c=i,h=o;if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,k)=>w>>h[k]);return new P(d,n.typeInfo)}if(re(i)){const c=o,h=i.map((d,w)=>d>>c);return new P(h,n.typeInfo)}if(re(o)){const c=i,h=o.map((d,w)=>c>>d);return new P(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new B(a>>l,u)}case">":if(re(i)&&re(o)){const a=i,l=o;if(a.length!==l.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const u=a.map((c,h)=>c>l[h]?1:0);return new P(u,n.typeInfo)}if(re(i)){const a=o,l=i.map((u,c)=>u>a?1:0);return new P(l,n.typeInfo)}if(re(o)){const a=i,l=o.map((u,c)=>a>u?1:0);return new P(l,r.typeInfo)}return new B(i>o?1:0,this.getTypeInfo("bool"));case"<":if(re(i)&&re(o)){const a=i,l=o;if(a.length!==l.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const u=a.map((c,h)=>c<l[h]?1:0);return new P(u,n.typeInfo)}if(re(i)){const a=o,l=i.map((u,c)=>u<a?1:0);return new P(l,n.typeInfo)}if(re(o)){const a=i,l=o.map((u,c)=>a<u?1:0);return new P(l,r.typeInfo)}return new B(i<o?1:0,this.getTypeInfo("bool"));case"==":if(re(i)&&re(o)){const a=i,l=o;if(a.length!==l.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const u=a.map((c,h)=>c===l[h]?1:0);return new P(u,n.typeInfo)}if(re(i)){const a=o,l=i.map((u,c)=>u==a?1:0);return new P(l,n.typeInfo)}if(re(o)){const a=i,l=o.map((u,c)=>a==u?1:0);return new P(l,r.typeInfo)}return new B(i===o?1:0,this.getTypeInfo("bool"));case"!=":if(re(i)&&re(o)){const a=i,l=o;if(a.length!==l.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const u=a.map((c,h)=>c!==l[h]?1:0);return new P(u,n.typeInfo)}if(re(i)){const a=o,l=i.map((u,c)=>u!==a?1:0);return new P(l,n.typeInfo)}if(re(o)){const a=i,l=o.map((u,c)=>a!==u?1:0);return new P(l,r.typeInfo)}return new B(i!==o?1:0,this.getTypeInfo("bool"));case">=":if(re(i)&&re(o)){const a=i,l=o;if(a.length!==l.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const u=a.map((c,h)=>c>=l[h]?1:0);return new P(u,n.typeInfo)}if(re(i)){const a=o,l=i.map((u,c)=>u>=a?1:0);return new P(l,n.typeInfo)}if(re(o)){const a=i,l=o.map((u,c)=>a>=u?1:0);return new P(l,r.typeInfo)}return new B(i>=o?1:0,this.getTypeInfo("bool"));case"<=":if(re(i)&&re(o)){const a=i,l=o;if(a.length!==l.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const u=a.map((c,h)=>c<=l[h]?1:0);return new P(u,n.typeInfo)}if(re(i)){const a=o,l=i.map((u,c)=>u<=a?1:0);return new P(l,n.typeInfo)}if(re(o)){const a=i,l=o.map((u,c)=>a<=u?1:0);return new P(l,r.typeInfo)}return new B(i<=o?1:0,this.getTypeInfo("bool"));case"&&":if(re(i)&&re(o)){const a=i,l=o;if(a.length!==l.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const u=a.map((c,h)=>c&&l[h]?1:0);return new P(u,n.typeInfo)}if(re(i)){const a=o,l=i.map((u,c)=>u&&a?1:0);return new P(l,n.typeInfo)}if(re(o)){const a=i,l=o.map((u,c)=>a&&u?1:0);return new P(l,r.typeInfo)}return new B(i&&o?1:0,this.getTypeInfo("bool"));case"||":if(re(i)&&re(o)){const a=i,l=o;if(a.length!==l.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const u=a.map((c,h)=>c||l[h]?1:0);return new P(u,n.typeInfo)}if(re(i)){const a=o,l=i.map((u,c)=>u||a?1:0);return new P(l,n.typeInfo)}if(re(o)){const a=i,l=o.map((u,c)=>a||u?1:0);return new P(l,r.typeInfo)}return new B(i||o?1:0,this.getTypeInfo("bool"))}return console.error(`Unknown operator ${e.operator}. Line ${e.line}`),null}_evalCall(e,t){if(e.cachedReturnValue!==null)return e.cachedReturnValue;const n=t.clone();n.currentFunctionName=e.name;const r=t.getFunction(e.name);if(!r)return e.isBuiltin?this._callBuiltinFunction(e,n):this.getTypeInfo(e.name)?this._evalCreate(e,t):(console.error(`Unknown function "${e.name}". Line ${e.line}`),null);for(let i=0;i<r.node.args.length;++i){const o=r.node.args[i],a=this.evalExpression(e.args[i],n);n.createVariable(o.name,a,o)}return this._execStatements(r.node.body,n)}_callBuiltinFunction(e,t){switch(e.name){case"all":return this.builtins.All(e,t);case"any":return this.builtins.Any(e,t);case"select":return this.builtins.Select(e,t);case"arrayLength":return this.builtins.ArrayLength(e,t);case"abs":return this.builtins.Abs(e,t);case"acos":return this.builtins.Acos(e,t);case"acosh":return this.builtins.Acosh(e,t);case"asin":return this.builtins.Asin(e,t);case"asinh":return this.builtins.Asinh(e,t);case"atan":return this.builtins.Atan(e,t);case"atanh":return this.builtins.Atanh(e,t);case"atan2":return this.builtins.Atan2(e,t);case"ceil":return this.builtins.Ceil(e,t);case"clamp":return this.builtins.Clamp(e,t);case"cos":return this.builtins.Cos(e,t);case"cosh":return this.builtins.Cosh(e,t);case"countLeadingZeros":return this.builtins.CountLeadingZeros(e,t);case"countOneBits":return this.builtins.CountOneBits(e,t);case"countTrailingZeros":return this.builtins.CountTrailingZeros(e,t);case"cross":return this.builtins.Cross(e,t);case"degrees":return this.builtins.Degrees(e,t);case"determinant":return this.builtins.Determinant(e,t);case"distance":return this.builtins.Distance(e,t);case"dot":return this.builtins.Dot(e,t);case"dot4U8Packed":return this.builtins.Dot4U8Packed(e,t);case"dot4I8Packed":return this.builtins.Dot4I8Packed(e,t);case"exp":return this.builtins.Exp(e,t);case"exp2":return this.builtins.Exp2(e,t);case"extractBits":return this.builtins.ExtractBits(e,t);case"faceForward":return this.builtins.FaceForward(e,t);case"firstLeadingBit":return this.builtins.FirstLeadingBit(e,t);case"firstTrailingBit":return this.builtins.FirstTrailingBit(e,t);case"floor":return this.builtins.Floor(e,t);case"fma":return this.builtins.Fma(e,t);case"fract":return this.builtins.Fract(e,t);case"frexp":return this.builtins.Frexp(e,t);case"insertBits":return this.builtins.InsertBits(e,t);case"inverseSqrt":return this.builtins.InverseSqrt(e,t);case"ldexp":return this.builtins.Ldexp(e,t);case"length":return this.builtins.Length(e,t);case"log":return this.builtins.Log(e,t);case"log2":return this.builtins.Log2(e,t);case"max":return this.builtins.Max(e,t);case"min":return this.builtins.Min(e,t);case"mix":return this.builtins.Mix(e,t);case"modf":return this.builtins.Modf(e,t);case"normalize":return this.builtins.Normalize(e,t);case"pow":return this.builtins.Pow(e,t);case"quantizeToF16":return this.builtins.QuantizeToF16(e,t);case"radians":return this.builtins.Radians(e,t);case"reflect":return this.builtins.Reflect(e,t);case"refract":return this.builtins.Refract(e,t);case"reverseBits":return this.builtins.ReverseBits(e,t);case"round":return this.builtins.Round(e,t);case"saturate":return this.builtins.Saturate(e,t);case"sign":return this.builtins.Sign(e,t);case"sin":return this.builtins.Sin(e,t);case"sinh":return this.builtins.Sinh(e,t);case"smoothStep":return this.builtins.SmoothStep(e,t);case"sqrt":return this.builtins.Sqrt(e,t);case"step":return this.builtins.Step(e,t);case"tan":return this.builtins.Tan(e,t);case"tanh":return this.builtins.Tanh(e,t);case"transpose":return this.builtins.Transpose(e,t);case"trunc":return this.builtins.Trunc(e,t);case"dpdx":return this.builtins.Dpdx(e,t);case"dpdxCoarse":return this.builtins.DpdxCoarse(e,t);case"dpdxFine":return this.builtins.DpdxFine(e,t);case"dpdy":return this.builtins.Dpdy(e,t);case"dpdyCoarse":return this.builtins.DpdyCoarse(e,t);case"dpdyFine":return this.builtins.DpdyFine(e,t);case"fwidth":return this.builtins.Fwidth(e,t);case"fwidthCoarse":return this.builtins.FwidthCoarse(e,t);case"fwidthFine":return this.builtins.FwidthFine(e,t);case"textureDimensions":return this.builtins.TextureDimensions(e,t);case"textureGather":return this.builtins.TextureGather(e,t);case"textureGatherCompare":return this.builtins.TextureGatherCompare(e,t);case"textureLoad":return this.builtins.TextureLoad(e,t);case"textureNumLayers":return this.builtins.TextureNumLayers(e,t);case"textureNumLevels":return this.builtins.TextureNumLevels(e,t);case"textureNumSamples":return this.builtins.TextureNumSamples(e,t);case"textureSample":return this.builtins.TextureSample(e,t);case"textureSampleBias":return this.builtins.TextureSampleBias(e,t);case"textureSampleCompare":return this.builtins.TextureSampleCompare(e,t);case"textureSampleCompareLevel":return this.builtins.TextureSampleCompareLevel(e,t);case"textureSampleGrad":return this.builtins.TextureSampleGrad(e,t);case"textureSampleLevel":return this.builtins.TextureSampleLevel(e,t);case"textureSampleBaseClampToEdge":return this.builtins.TextureSampleBaseClampToEdge(e,t);case"textureStore":return this.builtins.TextureStore(e,t);case"atomicLoad":return this.builtins.AtomicLoad(e,t);case"atomicStore":return this.builtins.AtomicStore(e,t);case"atomicAdd":return this.builtins.AtomicAdd(e,t);case"atomicSub":return this.builtins.AtomicSub(e,t);case"atomicMax":return this.builtins.AtomicMax(e,t);case"atomicMin":return this.builtins.AtomicMin(e,t);case"atomicAnd":return this.builtins.AtomicAnd(e,t);case"atomicOr":return this.builtins.AtomicOr(e,t);case"atomicXor":return this.builtins.AtomicXor(e,t);case"atomicExchange":return this.builtins.AtomicExchange(e,t);case"atomicCompareExchangeWeak":return this.builtins.AtomicCompareExchangeWeak(e,t);case"pack4x8snorm":return this.builtins.Pack4x8snorm(e,t);case"pack4x8unorm":return this.builtins.Pack4x8unorm(e,t);case"pack4xI8":return this.builtins.Pack4xI8(e,t);case"pack4xU8":return this.builtins.Pack4xU8(e,t);case"pack4x8Clamp":return this.builtins.Pack4x8Clamp(e,t);case"pack4xU8Clamp":return this.builtins.Pack4xU8Clamp(e,t);case"pack2x16snorm":return this.builtins.Pack2x16snorm(e,t);case"pack2x16unorm":return this.builtins.Pack2x16unorm(e,t);case"pack2x16float":return this.builtins.Pack2x16float(e,t);case"unpack4x8snorm":return this.builtins.Unpack4x8snorm(e,t);case"unpack4x8unorm":return this.builtins.Unpack4x8unorm(e,t);case"unpack4xI8":return this.builtins.Unpack4xI8(e,t);case"unpack4xU8":return this.builtins.Unpack4xU8(e,t);case"unpack2x16snorm":return this.builtins.Unpack2x16snorm(e,t);case"unpack2x16unorm":return this.builtins.Unpack2x16unorm(e,t);case"unpack2x16float":return this.builtins.Unpack2x16float(e,t);case"storageBarrier":return this.builtins.StorageBarrier(e,t);case"textureBarrier":return this.builtins.TextureBarrier(e,t);case"workgroupBarrier":return this.builtins.WorkgroupBarrier(e,t);case"workgroupUniformLoad":return this.builtins.WorkgroupUniformLoad(e,t);case"subgroupAdd":return this.builtins.SubgroupAdd(e,t);case"subgroupExclusiveAdd":return this.builtins.SubgroupExclusiveAdd(e,t);case"subgroupInclusiveAdd":return this.builtins.SubgroupInclusiveAdd(e,t);case"subgroupAll":return this.builtins.SubgroupAll(e,t);case"subgroupAnd":return this.builtins.SubgroupAnd(e,t);case"subgroupAny":return this.builtins.SubgroupAny(e,t);case"subgroupBallot":return this.builtins.SubgroupBallot(e,t);case"subgroupBroadcast":return this.builtins.SubgroupBroadcast(e,t);case"subgroupBroadcastFirst":return this.builtins.SubgroupBroadcastFirst(e,t);case"subgroupElect":return this.builtins.SubgroupElect(e,t);case"subgroupMax":return this.builtins.SubgroupMax(e,t);case"subgroupMin":return this.builtins.SubgroupMin(e,t);case"subgroupMul":return this.builtins.SubgroupMul(e,t);case"subgroupExclusiveMul":return this.builtins.SubgroupExclusiveMul(e,t);case"subgroupInclusiveMul":return this.builtins.SubgroupInclusiveMul(e,t);case"subgroupOr":return this.builtins.SubgroupOr(e,t);case"subgroupShuffle":return this.builtins.SubgroupShuffle(e,t);case"subgroupShuffleDown":return this.builtins.SubgroupShuffleDown(e,t);case"subgroupShuffleUp":return this.builtins.SubgroupShuffleUp(e,t);case"subgroupShuffleXor":return this.builtins.SubgroupShuffleXor(e,t);case"subgroupXor":return this.builtins.SubgroupXor(e,t);case"quadBroadcast":return this.builtins.QuadBroadcast(e,t);case"quadSwapDiagonal":return this.builtins.QuadSwapDiagonal(e,t);case"quadSwapX":return this.builtins.QuadSwapX(e,t);case"quadSwapY":return this.builtins.QuadSwapY(e,t)}const n=t.getFunction(e.name);if(n){const r=t.clone();for(let i=0;i<n.node.args.length;++i){const o=n.node.args[i],a=this.evalExpression(e.args[i],r);r.setVariable(o.name,a,o)}return this._execStatements(n.node.body,r)}return null}_callConstructorValue(e,t){if(!e.args||e.args.length===0)return new B(0,this.getTypeInfo(e.type));const n=this.evalExpression(e.args[0],t);return n.typeInfo=this.getTypeInfo(e.type),n.getSubData(this,e.postfix,t).clone()}_callConstructorVec(e,t){const n=this.getTypeInfo(e.type),r=e.type.getTypeName(),i=bu[r];if(i===void 0)return console.error(`Invalid vec constructor ${r}. Line ${e.line}`),null;const o=[];if(e instanceof et)if(e.isVector){const a=e.vectorValue;for(const l of a)o.push(l)}else o.push(e.scalarValue);else if(e.args)for(const a of e.args){const l=this.evalExpression(a,t);if(l instanceof P){const u=l.data;for(let c=0;c<u.length;++c){let h=u[c];o.push(h)}}else if(l instanceof B){let u=l.value;o.push(u)}}if(e.type instanceof z&&e.type.format===null&&(e.type.format=z.f32),o.length===0){const a=Array(i).fill(0);return new P(a,n).getSubData(this,e.postfix,t)}if(o.length===1)for(;o.length<i;)o.push(o[0]);return o.length<i?(console.error("Invalid vec constructor. Line "+e.line),null):new P(o.length>i?o.slice(0,i):o,n).getSubData(this,e.postfix,t)}_callConstructorMatrix(e,t){const n=this.getTypeInfo(e.type),r=e.type.getTypeName(),i=It[r];if(i===void 0)return console.error(`Invalid matrix constructor ${r}. Line ${e.line}`),null;const o=[];if(e instanceof et)if(e.isVector){const a=e.vectorValue;for(const l of a)o.push(l)}else o.push(e.scalarValue);else if(e.args)for(const a of e.args){const l=this.evalExpression(a,t);l instanceof P?o.push(...l.data):l instanceof B?o.push(l.value):l instanceof de&&o.push(...l.data)}if(n instanceof Ar&&n.format===null&&(n.format=this.getTypeInfo("f32")),o.length===0){const a=Array(i[2]).fill(0);return new de(a,n).getSubData(this,e.postfix,t)}return o.length!==i[2]?(console.error("Invalid matrix constructor. Line "+e.line),null):new de(o,n).getSubData(this,e.postfix,t)}}_t._breakObj=new un(new an("BREAK",null),null),_t._continueObj=new un(new an("CONTINUE",null),null),_t._priority=new Map([["f32",0],["f16",1],["u32",2],["i32",3],["x32",3]]);class jA{constructor(){this.constants=new Map,this.aliases=new Map,this.structs=new Map}}class KA{constructor(){this._tokens=[],this._current=0,this._currentLine=1,this._deferArrayCountEval=[],this._currentLoop=[],this._context=new jA,this._exec=new _t,this._forwardTypeCount=0}parse(e){this._initialize(e),this._deferArrayCountEval.length=0;const t=[];for(;!this._isAtEnd();){const n=this._global_decl_or_directive();if(!n)break;t.push(n)}if(this._deferArrayCountEval.length>0){for(const n of this._deferArrayCountEval){const r=n.arrayType,i=n.countNode;if(i instanceof Yt){const o=i.name,a=this._context.constants.get(o);if(a)try{const l=a.constEvaluate(this._exec);r.count=l}catch{}}}this._deferArrayCountEval.length=0}if(this._forwardTypeCount>0)for(const n of t)n.search(r=>{r instanceof Id||r instanceof Ta?r.type=this._forwardType(r.type):r instanceof ao?r.format=this._forwardType(r.format):r instanceof ms||r instanceof oo||r instanceof Ea?r.type=this._forwardType(r.type):r instanceof Eo?r.returnType=this._forwardType(r.returnType):r instanceof kd&&(r.type=this._forwardType(r.type))});return t}_forwardType(e){if(e instanceof Sd){const t=this._getType(e.name);if(t)return t}else e instanceof Ta?e.type=this._forwardType(e.type):e instanceof ao&&(e.format=this._forwardType(e.format));return e}_initialize(e){if(e)if(typeof e=="string"){const t=new LA(e);this._tokens=t.scanTokens()}else this._tokens=e;else this._tokens=[];this._current=0}_updateNode(e,t){return e.line=t??this._currentLine,e}_error(e,t){return{token:e,message:t,toString:()=>""+t}}_isAtEnd(){return this._current>=this._tokens.length||this._peek().type==O.eof}_match(e){if(e instanceof U)return!!this._check(e)&&(this._advance(),!0);for(let t=0,n=e.length;t<n;++t){const r=e[t];if(this._check(r))return this._advance(),!0}return!1}_consume(e,t){if(this._check(e))return this._advance();throw this._error(this._peek(),`${t}. Line:${this._currentLine}`)}_check(e){if(this._isAtEnd())return!1;const t=this._peek();if(e instanceof Array){const n=t.type;let r=!1;for(const i of e){if(n===i)return!0;i===O.tokens.name&&(r=!0)}if(r){const i=O.tokens.name.rule.exec(t.lexeme);if(i&&i.index==0&&i[0]==t.lexeme)return!0}return!1}if(t.type===e)return!0;if(e===O.tokens.name){const n=O.tokens.name.rule.exec(t.lexeme);return n&&n.index==0&&n[0]==t.lexeme}return!1}_advance(){var e,t;return this._currentLine=(t=(e=this._peek())===null||e===void 0?void 0:e.line)!==null&&t!==void 0?t:-1,this._isAtEnd()||this._current++,this._previous()}_peek(){return this._tokens[this._current]}_previous(){return this._tokens[this._current-1]}_global_decl_or_directive(){for(;this._match(O.tokens.semicolon)&&!this._isAtEnd(););if(this._match(O.keywords.alias)){const t=this._type_alias();return this._consume(O.tokens.semicolon,"Expected ';'"),this._exec.reflection.updateAST([t]),t}if(this._match(O.keywords.diagnostic)){const t=this._diagnostic();return this._consume(O.tokens.semicolon,"Expected ';'"),this._exec.reflection.updateAST([t]),t}if(this._match(O.keywords.requires)){const t=this._requires_directive();return this._consume(O.tokens.semicolon,"Expected ';'"),this._exec.reflection.updateAST([t]),t}if(this._match(O.keywords.enable)){const t=this._enable_directive();return this._consume(O.tokens.semicolon,"Expected ';'"),this._exec.reflection.updateAST([t]),t}const e=this._attribute();if(this._check(O.keywords.var)){const t=this._global_variable_decl();return t!=null&&(t.attributes=e),this._consume(O.tokens.semicolon,"Expected ';'."),this._exec.reflection.updateAST([t]),t}if(this._check(O.keywords.override)){const t=this._override_variable_decl();return t!=null&&(t.attributes=e),this._consume(O.tokens.semicolon,"Expected ';'."),this._exec.reflection.updateAST([t]),t}if(this._check(O.keywords.let)){const t=this._global_let_decl();return t!=null&&(t.attributes=e),this._consume(O.tokens.semicolon,"Expected ';'."),this._exec.reflection.updateAST([t]),t}if(this._check(O.keywords.const)){const t=this._global_const_decl();return t!=null&&(t.attributes=e),this._consume(O.tokens.semicolon,"Expected ';'."),this._exec.reflection.updateAST([t]),t}if(this._check(O.keywords.struct)){const t=this._struct_decl();return t!=null&&(t.attributes=e),this._exec.reflection.updateAST([t]),t}if(this._check(O.keywords.fn)){const t=this._function_decl();return t!=null&&(t.attributes=e),this._exec.reflection.updateAST([t]),t}return null}_function_decl(){if(!this._match(O.keywords.fn))return null;const e=this._currentLine,t=this._consume(O.tokens.ident,"Expected function name.").toString();this._consume(O.tokens.paren_left,"Expected '(' for function arguments.");const n=[];if(!this._check(O.tokens.paren_right))do{if(this._check(O.tokens.paren_right))break;const a=this._attribute(),l=this._consume(O.tokens.name,"Expected argument name.").toString();this._consume(O.tokens.colon,"Expected ':' for argument type.");const u=this._attribute(),c=this._type_decl();c!=null&&(c.attributes=u,n.push(this._updateNode(new kd(l,c,a))))}while(this._match(O.tokens.comma));this._consume(O.tokens.paren_right,"Expected ')' after function arguments.");let r=null;if(this._match(O.tokens.arrow)){const a=this._attribute();r=this._type_decl(),r!=null&&(r.attributes=a)}const i=this._compound_statement(),o=this._currentLine;return this._updateNode(new Eo(t,n,r,i,e,o),e)}_compound_statement(){const e=[];for(this._consume(O.tokens.brace_left,"Expected '{' for block.");!this._check(O.tokens.brace_right);){const t=this._statement();t!==null&&e.push(t)}return this._consume(O.tokens.brace_right,"Expected '}' for block."),e}_statement(){for(;this._match(O.tokens.semicolon)&&!this._isAtEnd(););if(this._check(O.tokens.attr)&&this._attribute(),this._check(O.keywords.if))return this._if_statement();if(this._check(O.keywords.switch))return this._switch_statement();if(this._check(O.keywords.loop))return this._loop_statement();if(this._check(O.keywords.for))return this._for_statement();if(this._check(O.keywords.while))return this._while_statement();if(this._check(O.keywords.continuing))return this._continuing_statement();if(this._check(O.keywords.static_assert))return this._static_assert_statement();if(this._check(O.tokens.brace_left))return this._compound_statement();let e=null;if(this._check(O.keywords.return))e=this._return_statement();else if(this._check([O.keywords.var,O.keywords.let,O.keywords.const]))e=this._variable_statement();else if(this._match(O.keywords.discard))e=this._updateNode(new MA);else if(this._match(O.keywords.break)){const t=this._updateNode(new Cy);if(this._currentLoop.length>0){const n=this._currentLoop[this._currentLoop.length-1];t.loopId=n.id}e=t,this._check(O.keywords.if)&&(this._advance(),t.condition=this._optional_paren_expression())}else if(this._match(O.keywords.continue)){const t=this._updateNode(new Ny);if(!(this._currentLoop.length>0))throw this._error(this._peek(),"Continue statement must be inside a loop. Line: "+t.line);{const n=this._currentLoop[this._currentLoop.length-1];t.loopId=n.id}e=t}else e=this._increment_decrement_statement()||this._func_call_statement()||this._assignment_statement();return e!=null&&this._consume(O.tokens.semicolon,"Expected ';' after statement."),e}_static_assert_statement(){if(!this._match(O.keywords.static_assert))return null;const e=this._currentLine,t=this._optional_paren_expression();return this._updateNode(new $A(t),e)}_while_statement(){if(!this._match(O.keywords.while))return null;const e=this._updateNode(new xy(null,null));return this._currentLoop.push(e),e.condition=this._optional_paren_expression(),this._check(O.tokens.attr)&&this._attribute(),e.body=this._compound_statement(),this._currentLoop.pop(),e}_continuing_statement(){const e=this._currentLoop.length>0?this._currentLoop[this._currentLoop.length-1].id:-1;if(!this._match(O.keywords.continuing))return null;const t=this._currentLine,n=this._compound_statement();return this._updateNode(new nc(n,e),t)}_for_statement(){if(!this._match(O.keywords.for))return null;this._consume(O.tokens.paren_left,"Expected '('.");const e=this._updateNode(new _y(null,null,null,null));return this._currentLoop.push(e),e.init=this._check(O.tokens.semicolon)?null:this._for_init(),this._consume(O.tokens.semicolon,"Expected ';'."),e.condition=this._check(O.tokens.semicolon)?null:this._short_circuit_or_expression(),this._consume(O.tokens.semicolon,"Expected ';'."),e.increment=this._check(O.tokens.paren_right)?null:this._for_increment(),this._consume(O.tokens.paren_right,"Expected ')'."),this._check(O.tokens.attr)&&this._attribute(),e.body=this._compound_statement(),this._currentLoop.pop(),e}_for_init(){return this._variable_statement()||this._func_call_statement()||this._assignment_statement()}_for_increment(){return this._func_call_statement()||this._increment_decrement_statement()||this._assignment_statement()}_variable_statement(){if(this._check(O.keywords.var)){const e=this._variable_decl();if(e===null)throw this._error(this._peek(),"Variable declaration expected.");let t=null;return this._match(O.tokens.equal)&&(t=this._short_circuit_or_expression()),this._updateNode(new ms(e.name,e.type,e.storage,e.access,t),e.line)}if(this._match(O.keywords.let)){const e=this._currentLine,t=this._consume(O.tokens.name,"Expected name for let.").toString();let n=null;if(this._match(O.tokens.colon)){const i=this._attribute();n=this._type_decl(),n!=null&&(n.attributes=i)}this._consume(O.tokens.equal,"Expected '=' for let.");const r=this._short_circuit_or_expression();return this._updateNode(new oo(t,n,null,null,r),e)}if(this._match(O.keywords.const)){const e=this._currentLine,t=this._consume(O.tokens.name,"Expected name for const.").toString();let n=null;if(this._match(O.tokens.colon)){const i=this._attribute();n=this._type_decl(),n!=null&&(n.attributes=i)}this._consume(O.tokens.equal,"Expected '=' for const.");const r=this._short_circuit_or_expression();return n===null&&r instanceof et&&(n=r.type),this._updateNode(new Ea(t,n,null,null,r),e)}return null}_increment_decrement_statement(){const e=this._current,t=this._unary_expression();if(t==null)return null;if(!this._check(O.increment_operators))return this._current=e,null;const n=this._consume(O.increment_operators,"Expected increment operator");return this._updateNode(new vy(n.type===O.tokens.plus_plus?Yr.increment:Yr.decrement,t))}_assignment_statement(){let e=null;const t=this._currentLine;if(this._check(O.tokens.brace_right))return null;let n=this._match(O.tokens.underscore);if(n||(e=this._unary_expression()),!n&&e==null)return null;const r=this._consume(O.assignment_operators,"Expected assignment operator."),i=this._short_circuit_or_expression();return this._updateNode(new Sy(Vi.parse(r.lexeme),e,i),t)}_func_call_statement(){if(!this._check(O.tokens.ident))return null;const e=this._currentLine,t=this._current,n=this._consume(O.tokens.ident,"Expected function name."),r=this._argument_expression_list();return r===null?(this._current=t,null):this._updateNode(new xh(n.lexeme,r),e)}_loop_statement(){if(!this._match(O.keywords.loop))return null;this._check(O.tokens.attr)&&this._attribute(),this._consume(O.tokens.brace_left,"Expected '{' for loop.");const e=this._updateNode(new ky([],null));this._currentLoop.push(e);let t=this._statement();for(;t!==null;){if(Array.isArray(t))for(let n of t)e.body.push(n);else e.body.push(t);if(t instanceof nc){e.continuing=t;break}t=this._statement()}return this._currentLoop.pop(),this._consume(O.tokens.brace_right,"Expected '}' for loop."),e}_switch_statement(){if(!this._match(O.keywords.switch))return null;const e=this._updateNode(new Iy(null,[]));if(this._currentLoop.push(e),e.condition=this._optional_paren_expression(),this._check(O.tokens.attr)&&this._attribute(),this._consume(O.tokens.brace_left,"Expected '{' for switch."),e.cases=this._switch_body(),e.cases==null||e.cases.length==0)throw this._error(this._previous(),"Expected 'case' or 'default'.");return this._consume(O.tokens.brace_right,"Expected '}' for switch."),this._currentLoop.pop(),e}_switch_body(){const e=[];let t=!1;for(;this._check([O.keywords.default,O.keywords.case]);){if(this._match(O.keywords.case)){const n=this._case_selectors();for(const i of n)if(i instanceof Aa){if(t)throw this._error(this._previous(),"Multiple default cases in switch statement.");t=!0;break}this._match(O.tokens.colon),this._check(O.tokens.attr)&&this._attribute(),this._consume(O.tokens.brace_left,"Exected '{' for switch case.");const r=this._case_body();this._consume(O.tokens.brace_right,"Exected '}' for switch case."),e.push(this._updateNode(new Py(n,r)))}if(this._match(O.keywords.default)){if(t)throw this._error(this._previous(),"Multiple default cases in switch statement.");this._match(O.tokens.colon),this._check(O.tokens.attr)&&this._attribute(),this._consume(O.tokens.brace_left,"Exected '{' for switch default.");const n=this._case_body();this._consume(O.tokens.brace_right,"Exected '}' for switch default."),e.push(this._updateNode(new Ry(n)))}}return e}_case_selectors(){const e=[];for(this._match(O.keywords.default)?e.push(this._updateNode(new Aa)):e.push(this._shift_expression());this._match(O.tokens.comma);)this._match(O.keywords.default)?e.push(this._updateNode(new Aa)):e.push(this._shift_expression());return e}_case_body(){if(this._match(O.keywords.fallthrough))return this._consume(O.tokens.semicolon,"Expected ';'"),[];let e=this._statement();if(e==null)return[];e instanceof Array||(e=[e]);const t=this._case_body();return t.length==0?e:[...e,t[0]]}_if_statement(){if(!this._match(O.keywords.if))return null;const e=this._currentLine,t=this._optional_paren_expression();this._check(O.tokens.attr)&&this._attribute();const n=this._compound_statement();let r=[];this._match_elseif()&&(this._check(O.tokens.attr)&&this._attribute(),r=this._elseif_statement(r));let i=null;return this._match(O.keywords.else)&&(this._check(O.tokens.attr)&&this._attribute(),i=this._compound_statement()),this._updateNode(new Ey(t,n,r,i),e)}_match_elseif(){return this._tokens[this._current].type===O.keywords.else&&this._tokens[this._current+1].type===O.keywords.if&&(this._advance(),this._advance(),!0)}_elseif_statement(e=[]){const t=this._optional_paren_expression(),n=this._compound_statement();return e.push(this._updateNode(new PA(t,n))),this._match_elseif()&&(this._check(O.tokens.attr)&&this._attribute(),this._elseif_statement(e)),e}_return_statement(){if(!this._match(O.keywords.return))return null;const e=this._short_circuit_or_expression();return this._updateNode(new Ty(e))}_short_circuit_or_expression(){let e=this._short_circuit_and_expr();for(;this._match(O.tokens.or_or);)e=this._updateNode(new bn(this._previous().toString(),e,this._short_circuit_and_expr()));return e}_short_circuit_and_expr(){let e=this._inclusive_or_expression();for(;this._match(O.tokens.and_and);)e=this._updateNode(new bn(this._previous().toString(),e,this._inclusive_or_expression()));return e}_inclusive_or_expression(){let e=this._exclusive_or_expression();for(;this._match(O.tokens.or);)e=this._updateNode(new bn(this._previous().toString(),e,this._exclusive_or_expression()));return e}_exclusive_or_expression(){let e=this._and_expression();for(;this._match(O.tokens.xor);)e=this._updateNode(new bn(this._previous().toString(),e,this._and_expression()));return e}_and_expression(){let e=this._equality_expression();for(;this._match(O.tokens.and);)e=this._updateNode(new bn(this._previous().toString(),e,this._equality_expression()));return e}_equality_expression(){const e=this._relational_expression();return this._match([O.tokens.equal_equal,O.tokens.not_equal])?this._updateNode(new bn(this._previous().toString(),e,this._relational_expression())):e}_relational_expression(){let e=this._shift_expression();for(;this._match([O.tokens.less_than,O.tokens.greater_than,O.tokens.less_than_equal,O.tokens.greater_than_equal]);)e=this._updateNode(new bn(this._previous().toString(),e,this._shift_expression()));return e}_shift_expression(){let e=this._additive_expression();for(;this._match([O.tokens.shift_left,O.tokens.shift_right]);)e=this._updateNode(new bn(this._previous().toString(),e,this._additive_expression()));return e}_additive_expression(){let e=this._multiplicative_expression();for(;this._match([O.tokens.plus,O.tokens.minus]);)e=this._updateNode(new bn(this._previous().toString(),e,this._multiplicative_expression()));return e}_multiplicative_expression(){let e=this._unary_expression();for(;this._match([O.tokens.star,O.tokens.forward_slash,O.tokens.modulo]);)e=this._updateNode(new bn(this._previous().toString(),e,this._unary_expression()));return e}_unary_expression(){return this._match([O.tokens.minus,O.tokens.bang,O.tokens.tilde,O.tokens.star,O.tokens.and])?this._updateNode(new Xe(this._previous().toString(),this._unary_expression())):this._singular_expression()}_singular_expression(){const e=this._primary_expression(),t=this._postfix_expression();return t&&(e.postfix=t),e}_postfix_expression(){if(this._match(O.tokens.bracket_left)){const e=this._short_circuit_or_expression();this._consume(O.tokens.bracket_right,"Expected ']'.");const t=this._updateNode(new gi(e)),n=this._postfix_expression();return n&&(t.postfix=n),t}if(this._match(O.tokens.period)){const e=this._consume(O.tokens.name,"Expected member name."),t=this._postfix_expression(),n=this._updateNode(new Cr(e.lexeme));return t&&(n.postfix=t),n}return null}_getStruct(e){return this._context.aliases.has(e)?this._context.aliases.get(e).type:this._context.structs.has(e)?this._context.structs.get(e):null}_getType(e){const t=this._getStruct(e);if(t!==null)return t;switch(e){case"void":return Y.void;case"bool":return Y.bool;case"i32":return Y.i32;case"u32":return Y.u32;case"f32":return Y.f32;case"f16":return Y.f16;case"vec2f":return z.vec2f;case"vec3f":return z.vec3f;case"vec4f":return z.vec4f;case"vec2i":return z.vec2i;case"vec3i":return z.vec3i;case"vec4i":return z.vec4i;case"vec2u":return z.vec2u;case"vec3u":return z.vec3u;case"vec4u":return z.vec4u;case"vec2h":return z.vec2h;case"vec3h":return z.vec3h;case"vec4h":return z.vec4h;case"mat2x2f":return z.mat2x2f;case"mat2x3f":return z.mat2x3f;case"mat2x4f":return z.mat2x4f;case"mat3x2f":return z.mat3x2f;case"mat3x3f":return z.mat3x3f;case"mat3x4f":return z.mat3x4f;case"mat4x2f":return z.mat4x2f;case"mat4x3f":return z.mat4x3f;case"mat4x4f":return z.mat4x4f;case"mat2x2h":return z.mat2x2h;case"mat2x3h":return z.mat2x3h;case"mat2x4h":return z.mat2x4h;case"mat3x2h":return z.mat3x2h;case"mat3x3h":return z.mat3x3h;case"mat3x4h":return z.mat3x4h;case"mat4x2h":return z.mat4x2h;case"mat4x3h":return z.mat4x3h;case"mat4x4h":return z.mat4x4h;case"mat2x2i":return z.mat2x2i;case"mat2x3i":return z.mat2x3i;case"mat2x4i":return z.mat2x4i;case"mat3x2i":return z.mat3x2i;case"mat3x3i":return z.mat3x3i;case"mat3x4i":return z.mat3x4i;case"mat4x2i":return z.mat4x2i;case"mat4x3i":return z.mat4x3i;case"mat4x4i":return z.mat4x4i;case"mat2x2u":return z.mat2x2u;case"mat2x3u":return z.mat2x3u;case"mat2x4u":return z.mat2x4u;case"mat3x2u":return z.mat3x2u;case"mat3x3u":return z.mat3x3u;case"mat3x4u":return z.mat3x4u;case"mat4x2u":return z.mat4x2u;case"mat4x3u":return z.mat4x3u;case"mat4x4u":return z.mat4x4u}return null}_validateTypeRange(e,t){if(t.name==="i32"){if(e<-2147483648||e>2147483647)throw this._error(this._previous(),`Value out of range for i32: ${e}. Line: ${this._currentLine}.`)}else if(t.name==="u32"&&(e<0||e>4294967295))throw this._error(this._previous(),`Value out of range for u32: ${e}. Line: ${this._currentLine}.`)}_primary_expression(){if(this._match(O.tokens.ident)){const n=this._previous().toString();if(this._check(O.tokens.paren_left)){const r=this._argument_expression_list(),i=this._getType(n);return i!==null?this._updateNode(new Hn(i,r)):this._updateNode(new vh(n,r))}if(this._context.constants.has(n)){const r=this._context.constants.get(n);return this._updateNode(new $y(n,r.value))}return this._updateNode(new Yt(n))}if(this._match(O.tokens.int_literal)){const n=this._previous().toString();let r=n.endsWith("i")||n.endsWith("i")?Y.i32:n.endsWith("u")||n.endsWith("U")?Y.u32:Y.x32;const i=parseInt(n);return this._validateTypeRange(i,r),this._updateNode(new et(new B(i,this._exec.getTypeInfo(r)),r))}if(this._match(O.tokens.uint_literal)){const n=parseInt(this._previous().toString());return this._validateTypeRange(n,Y.u32),this._updateNode(new et(new B(n,this._exec.getTypeInfo(Y.u32)),Y.u32))}if(this._match([O.tokens.decimal_float_literal,O.tokens.hex_float_literal])){let n=this._previous().toString(),r=n.endsWith("h");r&&(n=n.substring(0,n.length-1));const i=parseFloat(n);this._validateTypeRange(i,r?Y.f16:Y.f32);const o=r?Y.f16:Y.f32;return this._updateNode(new et(new B(i,this._exec.getTypeInfo(o)),o))}if(this._match([O.keywords.true,O.keywords.false])){let n=this._previous().toString()===O.keywords.true.rule;return this._updateNode(new et(new B(n?1:0,this._exec.getTypeInfo(Y.bool)),Y.bool))}if(this._check(O.tokens.paren_left))return this._paren_expression();if(this._match(O.keywords.bitcast)){this._consume(O.tokens.less_than,"Expected '<'.");const n=this._type_decl();this._consume(O.tokens.greater_than,"Expected '>'.");const r=this._paren_expression();return this._updateNode(new Dy(n,r))}const e=this._type_decl(),t=this._argument_expression_list();return this._updateNode(new Hn(e,t))}_argument_expression_list(){if(!this._match(O.tokens.paren_left))return null;const e=[];do{if(this._check(O.tokens.paren_right))break;const t=this._short_circuit_or_expression();e.push(t)}while(this._match(O.tokens.comma));return this._consume(O.tokens.paren_right,"Expected ')' for agument list"),e}_optional_paren_expression(){this._match(O.tokens.paren_left);const e=this._short_circuit_or_expression();return this._match(O.tokens.paren_right),e}_paren_expression(){this._consume(O.tokens.paren_left,"Expected '('.");const e=this._short_circuit_or_expression();return this._consume(O.tokens.paren_right,"Expected ')'."),e}_struct_decl(){if(!this._match(O.keywords.struct))return null;const e=this._currentLine,t=this._consume(O.tokens.ident,"Expected name for struct.").toString();this._consume(O.tokens.brace_left,"Expected '{' for struct body.");const n=[];for(;!this._check(O.tokens.brace_right);){const o=this._attribute(),a=this._consume(O.tokens.name,"Expected variable name.").toString();this._consume(O.tokens.colon,"Expected ':' for struct member type.");const l=this._attribute(),u=this._type_decl();u!=null&&(u.attributes=l),this._check(O.tokens.brace_right)?this._match(O.tokens.comma):this._consume(O.tokens.comma,"Expected ',' for struct member."),n.push(this._updateNode(new Id(a,u,o)))}this._consume(O.tokens.brace_right,"Expected '}' after struct body.");const r=this._currentLine,i=this._updateNode(new cs(t,n,e,r),e);return this._context.structs.set(t,i),i}_global_variable_decl(){const e=this._variable_decl();if(!e)return null;if(this._match(O.tokens.equal)){const t=this._const_expression();e.value=t}if(e.type!==null&&e.value instanceof et){if(e.value.type.name!=="x32"&&e.type.getTypeName()!==e.value.type.getTypeName())throw this._error(this._peek(),`Invalid cast from ${e.value.type.name} to ${e.type.name}. Line:${this._currentLine}`);e.value.isScalar&&this._validateTypeRange(e.value.scalarValue,e.type),e.value.type=e.type}else e.type===null&&e.value instanceof et&&(e.type=e.value.type.name==="x32"?Y.i32:e.value.type,e.value.isScalar&&this._validateTypeRange(e.value.scalarValue,e.type));return e}_override_variable_decl(){const e=this._override_decl();return e&&this._match(O.tokens.equal)&&(e.value=this._const_expression()),e}_global_const_decl(){var e;if(!this._match(O.keywords.const))return null;const t=this._consume(O.tokens.name,"Expected variable name"),n=this._currentLine;let r=null;if(this._match(O.tokens.colon)){const l=this._attribute();r=this._type_decl(),r!=null&&(r.attributes=l)}let i=null;this._consume(O.tokens.equal,"const declarations require an assignment");const o=this._short_circuit_or_expression();try{let l=[Y.f32],u=o.constEvaluate(this._exec,l);u instanceof B&&this._validateTypeRange(u.value,l[0]),l[0]instanceof z&&l[0].format===null&&u.typeInfo instanceof Ar&&u.typeInfo.format!==null&&(u.typeInfo.format.name==="f16"?l[0].format=Y.f16:u.typeInfo.format.name==="f32"?l[0].format=Y.f32:u.typeInfo.format.name==="i32"?l[0].format=Y.i32:u.typeInfo.format.name==="u32"?l[0].format=Y.u32:u.typeInfo.format.name==="bool"?l[0].format=Y.bool:console.error("TODO: impelement template format type "+u.typeInfo.format.name)),i=this._updateNode(new et(u,l[0])),this._exec.context.setVariable(t.toString(),u)}catch{i=o}if(r!==null&&i instanceof et){if(i.type.name!=="x32"&&r.getTypeName()!==i.type.getTypeName())throw this._error(this._peek(),`Invalid cast from ${i.type.name} to ${r.name}. Line:${this._currentLine}`);i.type=r,i.isScalar&&this._validateTypeRange(i.scalarValue,i.type)}else r===null&&i instanceof et&&(r=(e=i?.type)!==null&&e!==void 0?e:Y.f32,r===Y.x32&&(r=Y.i32));const a=this._updateNode(new Ea(t.toString(),r,"","",i),n);return this._context.constants.set(a.name,a),a}_global_let_decl(){if(!this._match(O.keywords.let))return null;const e=this._currentLine,t=this._consume(O.tokens.name,"Expected variable name");let n=null;if(this._match(O.tokens.colon)){const i=this._attribute();n=this._type_decl(),n!=null&&(n.attributes=i)}let r=null;if(this._match(O.tokens.equal)&&(r=this._const_expression()),n!==null&&r instanceof et){if(r.type.name!=="x32"&&n.getTypeName()!==r.type.getTypeName())throw this._error(this._peek(),`Invalid cast from ${r.type.name} to ${n.name}. Line:${this._currentLine}`);r.type=n}else n===null&&r instanceof et&&(n=r.type.name==="x32"?Y.i32:r.type);return r instanceof et&&r.isScalar&&this._validateTypeRange(r.scalarValue,n),this._updateNode(new oo(t.toString(),n,"","",r),e)}_const_expression(){return this._short_circuit_or_expression()}_variable_decl(){if(!this._match(O.keywords.var))return null;const e=this._currentLine;let t="",n="";this._match(O.tokens.less_than)&&(t=this._consume(O.storage_class,"Expected storage_class.").toString(),this._match(O.tokens.comma)&&(n=this._consume(O.access_mode,"Expected access_mode.").toString()),this._consume(O.tokens.greater_than,"Expected '>'."));const r=this._consume(O.tokens.name,"Expected variable name");let i=null;if(this._match(O.tokens.colon)){const o=this._attribute();i=this._type_decl(),i!=null&&(i.attributes=o)}return this._updateNode(new ms(r.toString(),i,t,n,null),e)}_override_decl(){if(!this._match(O.keywords.override))return null;const e=this._consume(O.tokens.name,"Expected variable name");let t=null;if(this._match(O.tokens.colon)){const n=this._attribute();t=this._type_decl(),t!=null&&(t.attributes=n)}return this._updateNode(new wh(e.toString(),t,null))}_diagnostic(){this._consume(O.tokens.paren_left,"Expected '('");const e=this._consume(O.tokens.ident,"Expected severity control name.");this._consume(O.tokens.comma,"Expected ','");let t=this._consume(O.tokens.ident,"Expected diagnostic rule name.").toString();return this._match(O.tokens.period)&&(t+="."+this._consume(O.tokens.ident,"Expected diagnostic message.").toString()),this._consume(O.tokens.paren_right,"Expected ')'"),this._updateNode(new Ay(e.toString(),t))}_enable_directive(){const e=this._consume(O.tokens.ident,"identity expected.");return this._updateNode(new DA(e.toString()))}_requires_directive(){const e=[this._consume(O.tokens.ident,"identity expected.").toString()];for(;this._match(O.tokens.comma);){const t=this._consume(O.tokens.ident,"identity expected.");e.push(t.toString())}return this._updateNode(new OA(e))}_type_alias(){const e=this._consume(O.tokens.ident,"identity expected.");this._consume(O.tokens.equal,"Expected '=' for type alias.");let t=this._type_decl();if(t===null)throw this._error(this._peek(),"Expected Type for Alias.");this._context.aliases.has(t.name)&&(t=this._context.aliases.get(t.name).type);const n=this._updateNode(new _h(e.toString(),t));return this._context.aliases.set(n.name,n),n}_type_decl(){if(this._check([O.tokens.ident,...O.texel_format,O.keywords.bool,O.keywords.f32,O.keywords.i32,O.keywords.u32])){const n=this._advance().toString();if(this._context.structs.has(n))return this._context.structs.get(n);if(this._context.aliases.has(n))return this._context.aliases.get(n).type;if(!this._getType(n)){const r=this._updateNode(new Sd(n));return this._forwardTypeCount++,r}return this._updateNode(new Y(n))}let e=this._texture_sampler_types();if(e)return e;if(this._check(O.template_types)){let n=this._advance().toString(),r=null,i=null;return this._match(O.tokens.less_than)&&(r=this._type_decl(),i=null,this._match(O.tokens.comma)&&(i=this._consume(O.access_mode,"Expected access_mode for pointer").toString()),this._consume(O.tokens.greater_than,"Expected '>' for type.")),this._updateNode(new z(n,r,i))}if(this._match(O.keywords.ptr)){let n=this._previous().toString();this._consume(O.tokens.less_than,"Expected '<' for pointer.");const r=this._consume(O.storage_class,"Expected storage_class for pointer");this._consume(O.tokens.comma,"Expected ',' for pointer.");const i=this._type_decl();let o=null;return this._match(O.tokens.comma)&&(o=this._consume(O.access_mode,"Expected access_mode for pointer").toString()),this._consume(O.tokens.greater_than,"Expected '>' for pointer."),this._updateNode(new Ta(n,r.toString(),i,o))}const t=this._attribute();if(this._match(O.keywords.array)){let n=null,r=-1;const i=this._previous();let o=null;if(this._match(O.tokens.less_than)){n=this._type_decl(),this._context.aliases.has(n.name)&&(n=this._context.aliases.get(n.name).type);let l="";if(this._match(O.tokens.comma)){o=this._shift_expression();try{l=o.constEvaluate(this._exec).toString(),o=null}catch{l="1"}}this._consume(O.tokens.greater_than,"Expected '>' for array."),r=l?parseInt(l):0}const a=this._updateNode(new ao(i.toString(),t,n,r));return o&&this._deferArrayCountEval.push({arrayType:a,countNode:o}),a}return null}_texture_sampler_types(){if(this._match(O.sampler_type))return this._updateNode(new qi(this._previous().toString(),null,null));if(this._match(O.depth_texture_type))return this._updateNode(new qi(this._previous().toString(),null,null));if(this._match(O.sampled_texture_type)||this._match(O.multisampled_texture_type)){const e=this._previous();this._consume(O.tokens.less_than,"Expected '<' for sampler type.");const t=this._type_decl();return this._consume(O.tokens.greater_than,"Expected '>' for sampler type."),this._updateNode(new qi(e.toString(),t,null))}if(this._match(O.storage_texture_type)){const e=this._previous();this._consume(O.tokens.less_than,"Expected '<' for sampler type.");const t=this._consume(O.texel_format,"Invalid texel format.").toString();this._consume(O.tokens.comma,"Expected ',' after texel format.");const n=this._consume(O.access_mode,"Expected access mode for storage texture type.").toString();return this._consume(O.tokens.greater_than,"Expected '>' for sampler type."),this._updateNode(new qi(e.toString(),t,n))}return null}_attribute(){let e=[];for(;this._match(O.tokens.attr);){const t=this._consume(O.attribute_name,"Expected attribute name"),n=this._updateNode(new Ly(t.toString(),null));if(this._match(O.tokens.paren_left)){if(n.value=this._consume(O.literal_or_ident,"Expected attribute value").toString(),this._check(O.tokens.comma)){this._advance();do{const r=this._consume(O.literal_or_ident,"Expected attribute value").toString();n.value instanceof Array||(n.value=[n.value]),n.value.push(r)}while(this._match(O.tokens.comma))}this._consume(O.tokens.paren_right,"Expected ')'")}e.push(n)}return e.length==0?null:e}}class By extends Zn{constructor(e){super(),e&&this.update(e)}update(e){const t=new KA().parse(e);this.updateAST(t)}}var Th="@vertex fn vertex()->@builtin(position)vec4f {return vec4f(0);}@fragment fn fragment()->@location(0)vec4f {return vec4f(0);}@compute @workgroup_size(1)fn compute(){}",Pi,Mt,mn,Ri,ha,Ad;let Fy=(Ad=class{constructor(s,e,t){ee(this,Ri),ee(this,Pi),Lt(this,"Active"),ee(this,Mt),Lt(this,"Device"),ee(this,mn,[]),Lt(this,"Reflect"),Lt(this,"GPUPipeline"),j(this,Pi,t),j(this,Mt,e),this.Active=!0,this.Device=s}CreatePipelineLabel(s){return T(this,Pi)&&s&&`${T(this,Pi)} ${s}`||""}CreatePipelineLayout(s,e){return e??(e=this.CreatePipelineLabel(T(this,Mt)+" Pipeline Layout")),s=lt(s),this.Device.createPipelineLayout({label:e,bindGroupLayouts:s})}CreateShaderModule(s,e,t,n){s||(s=Th,Bt(ie.SHADER_CODE_NOT_FOUND)),e??(e=this.CreatePipelineLabel("Shader Module"));const r=Array.isArray(s)&&s.join(`

`)||s;return this.Reflect=new By(r),this.Device.createShaderModule({label:e,code:r,sourceMap:t,compilationHints:n})}CreateBuffer(s){const e=s.label??this.CreatePipelineLabel("Buffer");return this.Device.createBuffer({label:e,...s})}CreateReadableBuffer(s){const e=typeof s=="number",t=Ct.READABLE|(!e&&s.usage||0);let n=e&&s;n||(n=s.size);const r=s?.label??"Readable Buffer";return this.CreateBuffer({label:r,size:n,...s,usage:t})}CreateWritableBuffer(s){const e=typeof s=="number",t=Ct.WRITABLE|(!e&&s.usage||0);let n=e&&s;n||(n=s.size);const r=s?.label??"Writable Buffer";return this.CreateBuffer({label:r,size:n,...s,usage:t})}CreateUniformBuffer(s,e){!this.Reflect&&ue(ie.SHADER_MODULE_NOT_FOUND,`\`${T(this,Mt)}Pipeline.CreateUniformBuffer\`.
            Use \`${T(this,Mt)}Pipeline.CreateShaderModule\` before creating a uniform buffer.`);const t=this.Reflect.uniforms.find(({name:o})=>s===o);!t&&ue(ie.UNIFORM_NOT_FOUND,`\`${s}\` in shader uniforms.`),s==="resolution"&&Bt(ie.INVALID_UNIFORM_NAME,`\`${s}\`.`);const n=e?.label??s+" Uniform Buffer",r=new ArrayBuffer(t.size),i=Ct.UNIFORM|e?.usage;return{buffer:this.CreateBuffer({label:n,size:r.byteLength,...e,usage:i}),[s]:se(this,Ri,ha).call(this,t,r)}}CreateStorageBuffer(s,e=1){!this.Reflect&&ue(ie.SHADER_MODULE_NOT_FOUND,`\`${T(this,Mt)}Pipeline.CreateStorageBuffer\`.
            Use \`${T(this,Mt)}Pipeline.CreateShaderModule\` before creating a storage buffer.`);const t=this.Reflect.storage.find(({name:h})=>s===h);!t&&ue(ie.STORAGE_NOT_FOUND,`\`${s}\` in shader bindings.`);const n=typeof e=="number",r=n&&e||e.length,i=Ct.STORAGE|(!n&&e.usage||0),o=!n&&e.label||s+" Storage Buffer",a=t.format.size*r,l=new ArrayBuffer(a),u=h=>(Object.keys(h).forEach(d=>{if(h[d].buffer instanceof ArrayBuffer){const w=h[d].constructor,k=a/w.BYTES_PER_ELEMENT;h[d]=new w(l,0,k)}else u(h[d])}),h),c=se(this,Ri,ha).call(this,t,l);return{buffer:this.CreateBuffer({label:o,size:a,...e,usage:i}),[s]:c.buffer instanceof ArrayBuffer?new c.constructor(l,0,r):u(c)}}WriteBuffer(s,e,t=0,n,r){this.Device.queue.writeBuffer(s,t,e,n,r)}GetBufferMinBindingSize(s){return!this.Reflect&&ue(ie.SHADER_MODULE_NOT_FOUND,`\`${T(this,Mt)}Pipeline.GetBufferMinBindingSize\`.
            Use \`${T(this,Mt)}Pipeline.CreateShaderModule\` before requesting buffer's min binding size.`),this.Reflect.getBindGroups().flat().find(({name:t})=>s===t)?.size??ue(ie.BINDING_NOT_FOUND,`\`${s}\` in shader bind groups.`)}CreateBindGroupEntries(s,e=0){return Array.isArray(s)&&s.map((t,n)=>({binding:e?.[n]??n,resource:t}))||[{binding:e,resource:s}]}CreateBindGroupLayout(s,e){return e??(e=this.CreatePipelineLabel("Bind Group Layout")),s=Array.isArray(s)&&s.map((t,n)=>({...t,binding:t.binding??n}))||[{...s,binding:s.binding??0}],this.Device.createBindGroupLayout({entries:s,label:e})}CreateBindGroup(s,e=0,t){return t??(t=this.CreatePipelineLabel("Bind Group")),typeof e=="number"&&(e=this.GPUPipeline?this.GPUPipeline.getBindGroupLayout(e):ue(ie.PIPELINE_NOT_FOUND,`${T(this,Mt)}Pipeline.
                    Use \`${T(this,Mt)}Stage.AddPipeline\` before creating a bind group.`)),this.Device.createBindGroup({entries:s,label:t,layout:e})}SetBindGroups(s,e){const t=Array.isArray(s),n=Array.isArray(e);e=(e=t&&n?e.map(r=>lt(r)):n&&e||e&&[e])&&e||[],j(this,mn,t&&s.map((r,i)=>({bindGroup:r,dynamicOffsets:e,active:!0}))||[{bindGroup:s,dynamicOffsets:e,active:!0}])}AddBindGroups(s,e){const t=Array.isArray(s),n=Array.isArray(e);return e=(e=t&&n?e.map(r=>lt(r)):n&&e||e&&[e])&&e||[],T(this,mn).push(...t&&s.map(r=>({bindGroup:r,dynamicOffsets:e,active:!0}))||[{bindGroup:s,dynamicOffsets:e,active:!0}])}SetActiveBindGroups(s){s=lt(s);for(let e=T(this,mn).length;e--;)T(this,mn)[e].active=s.includes(e)}UseBindGroups(s){for(let e=0,t=0,n=T(this,mn).length;e<n;++e){const{bindGroup:r,dynamicOffsets:i,active:o}=T(this,mn)[e];o&&s.setBindGroup(t++,r,i)}}GetBindGroupsInfo(){!this.Reflect&&ue(ie.SHADER_MODULE_NOT_FOUND,`\`${T(this,Mt)}Pipeline.GetBindGroupsInfo\`.
            Use \`${T(this,Mt)}Pipeline.CreateShaderModule\` before requesting bind groups information.`);const s=T(this,mn).length,e=Array(s),t=this.Reflect.getBindGroups();for(let n=0;n<s;++n){const{bindGroup:{label:r},dynamicOffsets:i,active:o}=T(this,mn)[n];e[n]={label:r,active:o,dynamicOffsets:i,bindings:t[n]}}return e}ClearBindGroups(){T(this,mn).splice(0)}Destroy(){this.ClearBindGroups()}},Pi=new WeakMap,Mt=new WeakMap,mn=new WeakMap,Ri=new WeakSet,ha=function(s,e,t=0,n=[]){const{format:r}=s.type,i=s.type.members??r?.members;let o=t+(s.offset??0);if(!i){const a=Qy((r??s.type).name),l=s.size/Jy(a);return new(e0(a))(e,o,l)}for(let a=0,l={},u=r?.isStruct&&s.count||1;a<u;++a)i.forEach(c=>l[c.name]=se(this,Ri,ha).call(this,c,e,o)),r!=null&&r.isStruct&&(o+=s.stride),n.push(l);return n.length===1&&n[0]||n},Ad);var xs,On,fa,Li,Mn,wu,Cd,Nd;let XA=(Nd=class extends Fy{constructor(s,e,t){super(s,"Render",t),ee(this,wu),Lt(this,"DestroyPassEncoder",!1),ee(this,xs,[]),ee(this,On),ee(this,fa),Lt(this,"TextureView"),ee(this,Li,[0,0,0,0]),ee(this,Mn,[0,void 0,void 0,void 0,void 0]),j(this,fa,e)}async Init(s={}){let e=jl(s),{vertex:t,fragment:n}=s;!e&&!t&&(e=this.CreateShaderModule()),e&&(t??(t=this.CreateVertexState(e)),n??(n=this.CreateFragmentState(e)));const r=s.label??this.CreatePipelineLabel("Render Pipeline"),i=s.layout??"auto";return this.GPUPipeline=await this.Device.createRenderPipelineAsync({label:r,layout:i,vertex:t,fragment:n,...s})}CreateBlendComponent(s="add",e="one",t="zero"){return{operation:s,srcFactor:e,dstFactor:t}}CreateColorTargetState(s=T(this,fa),e,t){return e&&(e={color:e.color??{},alpha:e.alpha??{}}),{format:s,blend:e,writeMask:t}}CreateMultisampleState(s=4,e,t){return{count:s,mask:e,alphaToCoverageEnabled:t}}CreateStencilFaceState(s,e,t,n){return{compare:s,failOp:e,depthFailOp:t,passOp:n}}CreateDepthStencilState(s="depth24plus",e=!0,t="less",n,r,i,o,a,l,u){return{format:s,depthWriteEnabled:e,depthCompare:t,stencilFront:n,stencilBack:r,stencilReadMask:i,stencilWriteMask:o,depthBias:a,depthBiasSlopeScale:l,depthBiasClamp:u}}CreateVertexState(s,e="vertex",t,n){return{module:s,entryPoint:e,buffers:t=lt(t),constants:n}}CreateFragmentState(s,e="fragment",t,n){return t??(t=this.CreateColorTargetState()),{module:s,entryPoint:e,targets:t=lt(t),constants:n}}CreateVertexBufferLayout(s,e,t="vertex"){!this.Reflect&&ue(ie.SHADER_MODULE_NOT_FOUND,"`RenderPipeline.CreateVertexBufferLayout`.\n            Call `RenderPipeline.CreateShaderModule` before creating a vertex layout or vertex buffer.");const{entry:{vertex:n}}=this.Reflect,r=n.find(({name:a})=>t===a);!r&&ue(ie.VERTEX_ENTRY_NOT_FOUND,`\`${t}\` in vertex shader entries.`);let i=[],o=0;for(let a=0,l=(s=lt(s)).length;a<l;++a){const u=s[a],c=typeof u=="string",h=c?u:u.name,d=r.inputs.find(({name:w})=>h===w);if(d){const w=c?Yy(d.type.size):u.format;i.push(se(this,wu,Cd).call(this,w,+d.location,o)),o+=Zy(w)}else ThrowWarning(ie.VERTEX_ATTRIBUTE_NOT_FOUND,`\`${h}\` in vertex shader inputs.`)}return{arrayStride:o,stepMode:e,attributes:i}}CreateVertexBuffer(s,e=1,t,n="vertex"){const r=typeof e=="number",i=!r&&e.label||"Vertex Buffer",o=Ct.VERTEX|(!r&&e.usage||0);if(s instanceof Float32Array)return this.CreateBuffer({label:i,size:s.byteLength,...e,usage:o});const a=this.CreateVertexBufferLayout(s,t,n),l=(r&&e||(e.count??1))*a.arrayStride;return{buffer:this.CreateBuffer({label:i,size:l,...e,usage:o}),layout:a}}SetVertexBuffers(s,e,t){t=lt(t),e=lt(e),j(this,xs,Array.isArray(s)&&s.map((n,r)=>({buffer:n,offset:e[r],size:t[r]}))||[{buffer:s,offset:e[0],size:t[0]}])}AddVertexBuffers(s,e,t){return t=lt(t),e=lt(e),T(this,xs).push(...Array.isArray(s)&&s.map((n,r)=>({buffer:n,offset:e[r],size:t[r]}))||[{buffer:s,offset:e[0],size:t[0]}])}CreateIndexBuffer(s,e){const t=Ct.INDEX|e?.usage,n=e?.label??"Index Buffer";return s=Array.isArray(s)&&new Uint32Array(s)||s,this.CreateBuffer({label:n,size:s.byteLength,...e,usage:t})}SetIndexBuffer(s,e="uint32",t,n){j(this,On,s&&{buffer:s,format:e,offset:t,size:n})}UseRenderBuffers(s){for(let e=0,t=T(this,xs).length;e<t;++e){const{buffer:n,offset:r,size:i}=T(this,xs)[e];s.setVertexBuffer(e,n,r,i)}T(this,On)&&s.setIndexBuffer(T(this,On).buffer,T(this,On).format,T(this,On).offset,T(this,On).size)}SetDrawParams(s,e,t,n,r){T(this,Mn)[0]=s,T(this,Mn)[1]=e,T(this,Mn)[2]=t,T(this,Mn)[3]=n,T(this,Mn)[4]=r,r!==void 0&&(T(this,Mn)[3]=r,T(this,Mn)[4]=n)}get ColorAttachment(){return 0}set BlendConstant(s){j(this,Li,t0(s))}get BlendConstant(){return T(this,Li)}get DrawMethod(){return T(this,On)?"drawIndexed":"draw"}get DrawParams(){return T(this,Mn)}Destroy(){var s;super.Destroy(),this.DestroyPassEncoder=!1,j(this,Li,[0,0,0,0]),T(this,xs).forEach(({buffer:e})=>e.destroy()),(s=T(this,On))==null||s.buffer.destroy(),T(this,xs).splice(0)}},xs=new WeakMap,On=new WeakMap,fa=new WeakMap,Li=new WeakMap,Mn=new WeakMap,wu=new WeakSet,Cd=function(s,e=0,t=0){return{format:s,shaderLocation:e,offset:t}},Nd),YA=class extends Fy{constructor(s,e){super(s,"Compute",e)}async Init(s){const e=s.label??this.CreatePipelineLabel("Compute Pipeline"),t=s.layout??"auto",n=jl(s)??this.CreateShaderModule();return this.GPUPipeline=await this.Device.createComputePipelineAsync({label:e,layout:t,compute:{module:n,...s}})}};var Zr,Ur,Ws,Gs,Ca,Na,Hi,or,sn,lo,zr,Uy;class zy{constructor(e,t,n){ee(this,sn),ee(this,Zr),ee(this,Ur),ee(this,Ws),Lt(this,"Device"),Lt(this,"BindGroups",[]),Lt(this,"Reflect"),ee(this,Gs),Lt(this,"Pipeline"),Lt(this,"Descriptor"),ee(this,Ca),ee(this,Na),ee(this,Hi),ee(this,or,[]),!e&&ue(ie.DEVICE_NOT_REQUESTED),j(this,Zr,n),this.Device=e,j(this,Ur,t),j(this,Ws,this.CreatePipelineLabel("Command Encoder"))}CreatePipelineLabel(e){return T(this,Ur)&&e&&`${T(this,Ur)} ${e}`||""}CreatePipelineLayout(e,t){t??(t=this.CreatePipelineLabel(T(this,Zr)+" Pipeline Layout"));const n=Array.isArray(e)&&e||[e];return this.Device.createPipelineLayout({label:t,bindGroupLayouts:n})}CreateTimestampWrites(e,t,n){return{querySet:e,beginningOfPassWriteIndex:t,endOfPassWriteIndex:n}}ResolveQuerySet(e,t,n=0,r=e.count,i=0){this.GetCommandEncoder(!0).resolveQuerySet(e,n,r,t,i)}CreateShaderModule(e,t,n,r){e||(e=Th,Bt(ie.SHADER_CODE_NOT_FOUND)),t??(t=this.CreatePipelineLabel("Shader Module"));const i=Array.isArray(e)&&e.join(`

`)||e;return this.Reflect=new By(i),this.Device.createShaderModule({label:t,code:i,sourceMap:n,compilationHints:r})}GetShaderModule(e){return e instanceof GPUShaderModule&&e||e.module}CreateBuffer(e){const t=e.label??this.CreatePipelineLabel("Buffer");return this.Device.createBuffer({label:t,...e})}CreateReadableBuffer(e){let t=typeof e=="number"&&e;t||(t=e.size);const n=e?.label??"Readable Buffer";return this.CreateBuffer({label:n,size:t,usage:Ct.READABLE,...e})}CreateWritableBuffer(e){let t=typeof e=="number"&&e;t||(t=e.size);const n=e?.label??"Writable Buffer";return this.CreateBuffer({label:n,size:t,usage:Ct.WRITABLE,...e})}CreateStorageBuffer(e,t=1){!this.Reflect&&ue(ie.SHADER_MODULE_NOT_FOUND,"`CreateStorageBuffer`.\n            Use `CreateShaderModule` before creating a storage buffer.");const n=this.Reflect.storage.find(({name:c})=>e===c);!n&&ue(ie.STORAGE_NOT_FOUND,`\`${e}\` in shader bindings.`);const r=typeof t=="number"&&t||t.length,i=t.label??e+" Storage Buffer",o=n.format.size*r,a=new ArrayBuffer(o),l=c=>(Object.keys(c).forEach(h=>{if(c[h].buffer instanceof ArrayBuffer){const d=c[h].constructor,w=o/d.BYTES_PER_ELEMENT;c[h]=new d(a,0,w)}else l(c[h])}),c),u=se(this,sn,lo).call(this,n,a);return{buffer:this.CreateBuffer({label:i,size:o,usage:Ct.STORAGE,...t}),[e]:u.buffer instanceof ArrayBuffer?new u.constructor(a,0,r):l(u)}}CreateUniformBuffer(e,t){!this.Reflect&&ue(ie.SHADER_MODULE_NOT_FOUND,"`CreateUniformBuffer`.\n            Use `CreateShaderModule` before creating a uniform buffer.");const n=this.Reflect.uniforms.find(({name:o})=>e===o);!n&&ue(ie.UNIFORM_NOT_FOUND,`\`${e}\` in shader uniforms.`),e==="resolution"&&Bt(ie.INVALID_UNIFORM_NAME,`\`${e}\`.`);const r=t?.label??e+" Uniform Buffer",i=new ArrayBuffer(n.size);return{buffer:this.CreateBuffer({label:r,size:i.byteLength,usage:Ct.UNIFORM,...t}),[e]:se(this,sn,lo).call(this,n,i)}}CreateUniformBufferLayout(e){!this.Reflect&&ue(ie.SHADER_MODULE_NOT_FOUND,"`CreateUniformBufferLayout`.\n            Use `CreateShaderModule` before creating a uniform buffer layout.");const t=this.Reflect.uniforms.find(({name:n})=>e===n);return!t&&ue(ie.UNIFORM_NOT_FOUND,`\`${e}\` in shader uniforms.`),e==="resolution"&&Bt(ie.INVALID_UNIFORM_NAME,`\`${e}\`.`),se(this,sn,lo).call(this,t,new ArrayBuffer(t.size))}WriteBuffer(e,t,n=0,r,i){this.Device.queue.writeBuffer(e,n,t,r,i)}CopyBufferToBuffer(e,t,n=t.size,r=0,i=0){this.GetCommandEncoder(!0).copyBufferToBuffer(e,r,t,i,n)}GetBufferMinBindingSize(e){return!this.Reflect&&ue(ie.SHADER_MODULE_NOT_FOUND,"`GetBufferMinBindingSize`.\n            Use `CreateShaderModule` before requesting buffer's min binding size."),this.Reflect.getBindGroups().flat().find(({name:n})=>e===n)?.size??ue(ie.BINDING_NOT_FOUND,`\`${e}\` in shader bind groups.`)}CreateBufferBindingLayout(e,t,n,r,i){return r??(r=se(this,sn,zr).call(this)),{binding:i,visibility:r,buffer:{type:e,hasDynamicOffset:t,minBindingSize:n}}}CreateSamplerBindingLayout(e,t,n){return t??(t=se(this,sn,zr).call(this)),{binding:n,visibility:t,sampler:{type:e}}}CreateTextureBindingLayout(e,t,n,r,i){return r??(r=se(this,sn,zr).call(this)),{binding:i,visibility:r,texture:{sampleType:e,viewDimension:t,multisampled:n}}}CreateStorageTextureBindingLayout(e,t,n,r,i){return r??(r=se(this,sn,zr).call(this)),{binding:i,visibility:r,storageTexture:{access:t,format:e,viewDimension:n}}}CreateExternalTextureBindingLayout(e,t){return e??(e=se(this,sn,zr).call(this)),{binding:t,visibility:e,externalTexture:{}}}CreateBindGroupEntries(e,t=0){return Array.isArray(e)&&e.map((n,r)=>({binding:t?.[r]??r,resource:n}))||[{binding:t,resource:e}]}CreateBindGroupLayout(e,t){return t??(t=this.CreatePipelineLabel("Bind Group Layout")),e=Array.isArray(e)&&e.map((n,r)=>({...n,binding:n.binding??r}))||[{...e,binding:e.binding??0}],this.Device.createBindGroupLayout({entries:e,label:t})}CreateBindGroup(e,t=0,n){return n??(n=this.CreatePipelineLabel("Bind Group")),typeof t=="number"&&(t=this.Pipeline?this.Pipeline.getBindGroupLayout(t):ue(ie.PIPELINE_NOT_FOUND,T(this,Zr)+"Pipeline.")),this.Device.createBindGroup({entries:e,label:n,layout:t})}SetBindGroups(e,t){const n=Array.isArray(e),r=Array.isArray(t);t=(t=n&&r?t.map(i=>Array.isArray(i)&&i||[i]):r&&t||t&&[t])&&t||[],this.BindGroups=n&&e.map((i,o)=>({bindGroup:i,dynamicOffsets:t,active:!0}))||[{bindGroup:e,dynamicOffsets:t,active:!0}]}AddBindGroups(e,t){const n=Array.isArray(e),r=Array.isArray(t);t=(t=n&&r?t.map(o=>Array.isArray(o)&&o||[o]):r&&t||t&&[t])&&t||[];const i=this.BindGroups.push(...n&&e.map(o=>({bindGroup:o,dynamicOffsets:t,active:!0}))||[{bindGroup:e,dynamicOffsets:t,active:!0}])-1;return!n&&[i]||Array.from({length:e.length}).map((o,a)=>i-a)}SetActiveBindGroups(e){e=Array.isArray(e)&&e||[e];for(let t=this.BindGroups.length;t--;)this.BindGroups[t].active=e.includes(t)}ClearBindGroups(){this.BindGroups.splice(0)}GetBindGroupsInfo(){!this.Reflect&&ue(ie.SHADER_MODULE_NOT_FOUND,"`GetBindGroupsInfo`.\n            Use `CreateShaderModule` before requesting bind groups information.");const e=this.BindGroups.length,t=Array(e),n=this.Reflect.getBindGroups();for(let r=0;r<e;++r){const{bindGroup:{label:i},dynamicOffsets:o,active:a}=this.BindGroups[r];t[r]={label:i,active:a,dynamicOffsets:o,bindings:n[r]}}return t}CreateCommandEncoder(){return j(this,Gs,this.Device.createCommandEncoder({label:T(this,Ws)}))}SetCommandEncoder(e){j(this,Gs,e)}GetCommandEncoder(e=!1){if(!T(this,Gs)){if(e){const t=""+(T(this,Ws)&&`Label: "${T(this,Ws)}".`);Bt(ie.COMMAND_ENCODER_NOT_FOUND,` ${t} Creating a new one.`)}return this.CreateCommandEncoder()}return T(this,Gs)}SubmitCommandBuffer(){this.Device.queue.submit([T(this,Gs).finish()])}SetPipeline(e){return this.Pipeline=e}SavePipelineState(){j(this,Hi,this.Reflect),j(this,Na,this.Pipeline),j(this,Ca,this.Descriptor),j(this,or,[...this.BindGroups])}ResetPipelineState(){this.ClearBindGroups()}RestorePipelineState(){this.Descriptor=T(this,Ca),this.SetPipeline(T(this,Na)),this.Reflect=T(this,Hi),se(this,sn,Uy).call(this)}set CommandEncoderLabel(e){j(this,Ws,e)}get ProgramName(){return T(this,Ur)}Destroy(){this.ResetPipelineState(),j(this,Hi,void 0),T(this,or).splice(0),this.SetCommandEncoder(void 0)}}Zr=new WeakMap,Ur=new WeakMap,Ws=new WeakMap,Gs=new WeakMap,Ca=new WeakMap,Na=new WeakMap,Hi=new WeakMap,or=new WeakMap,sn=new WeakSet,lo=function(s,e,t=0,n=[]){const{format:r}=s.type,i=s.type.members??r?.members;let o=t+(s.offset??0);if(!i){const a=Qy((r??s.type).name),l=s.size/Jy(a);return new(e0(a))(e,o,l)}for(let a=0,l={},u=r?.isStruct&&s.count||1;a<u;++a)i.forEach(c=>l[c.name]=se(this,sn,lo).call(this,c,e,o)),r!=null&&r.isStruct&&(o+=s.stride),n.push(l);return n.length===1&&n[0]||n},zr=function(){return T(this,Zr)==="Render"&&GPUShaderStage.FRAGMENT||GPUShaderStage.COMPUTE},Uy=function(){const s=T(this,or).map(({bindGroup:r})=>r),e=T(this,or).map(({dynamicOffsets:r})=>r),t=e.some(r=>typeof r=="number")&&e||void 0,n=T(this,or).map(({active:r},i)=>r&&i).filter(r=>typeof r=="number");this.SetBindGroups(s,t),this.SetActiveBindGroups(n)};const To=Ue({RENDER:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST,STORAGE:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING}),ZA=Ue({ALL:"all",STENCIL:"stencil-only",DEPTH:"depth-only"}),QA=Ue({CLAMP:"clamp-to-edge",REPEAT:"repeat",MIRROR:"mirror-repeat"}),Ao=Ue({NEAREST:"nearest",LINEAR:"linear"});Object.freeze(Object.defineProperty({__proto__:null,ADDRESS:QA,ASPECT:ZA,COMPARE:Ue({NEVER:"never",LESS:"less",EQUAL:"equal",LESS_EQUAL:"less-equal",GREATER:"greater",NOT_EQUAL:"not-equal",GREATER_EQUAL:"greater-equal",ALWAYS:"always"}),FILTER:Ao,USAGE:To},Symbol.toStringTag,{value:"Module"}));var Ah="const QUAD=array(vec2f(-1.0,-1.0),vec2f(1.0,-1.0),vec2f(1.0,1.0),vec2f(1.0,1.0),vec2f(-1.0,1.0),vec2f(-1.0,-1.0));fn GetQuadCoord(index: u32)->vec2f{return QUAD[index];}struct VertexOutput{@builtin(position)position: vec4f,@location(0)textureCoord: vec2f};@group(0)@binding(0)var Sampler: sampler;@group(0)@binding(1)var Texture: texture_2d<f32>;@vertex fn vertex(@builtin(vertex_index)index: u32)->VertexOutput {let position=GetQuadCoord(index);let coord=(position+1)*0.5;var output: VertexOutput;output.position=vec4f(position,0.0,1.0);output.textureCoord=vec2f(coord.x,1-coord.y);return output;}@fragment fn fragment(@location(0)textureCoord: vec2f)->@location(0)vec4f {return textureSample(Texture,Sampler,textureCoord);}",Wy="enable dual_source_blending;struct DSBFragmentOutput{@location(0)@blend_src(0)source: vec4f,@location(0)@blend_src(1)factor: vec4f};@fragment fn dsbTextFragment(input: TextVertexOutput)->DSBFragmentOutput {var output: DSBFragmentOutput;let coverage=GetSubpixelCoverage(input.inverseTextureSize,input.distanceDelta,input.fontUV);output.source=Font.color;output.factor=vec4f(coverage,Font.color.a);return output;}",Gy="override TRIPLET_FACTOR=0.6;const THRESHOLD=20.0/256.0;const MIN_GRAD=THRESHOLD*0.1;struct font{color: vec4f,back: vec4f,subpx: f32,hint: f32};struct text{matrix: mat3x3f,textureSize: vec2f};struct TextVertexOutput{@location(0)fontUV: vec2f,@location(1)screenUV: vec2f,@location(2)distanceDelta: f32,@builtin(position)position: vec4f,@location(3)inverseTextureSize: vec2f};@group(0)@binding(0)var Sampler: sampler;@group(0)@binding(1)var<uniform>Text: text;@group(0)@binding(2)var<uniform>Font: font;@group(0)@binding(3)var Texture: texture_2d<f32>;@vertex fn textVertex(@location(0)position: vec2f,@location(1)texture: vec2f,@location(2)size: f32)->TextVertexOutput{var output: TextVertexOutput;let clipSpace=Text.matrix*vec3f(position,1);output.inverseTextureSize=1.0/Text.textureSize;output.position=vec4f(clipSpace.xy,0,1);output.distanceDelta=1.0/size;output.screenUV=clipSpace.xy;output.fontUV=texture;return output;}fn GetSubpixelCoverage(size: vec2f,distance: f32,uv: vec2f)->vec3f{let sdf=textureSample(Texture,Sampler,uv).r;let sdfX=textureSample(Texture,Sampler,uv+vec2f(size.x,0)).r;let sdfY=textureSample(Texture,Sampler,uv+vec2f(0,size.y)).r;let strokeGradient=vec2f(sdfX-sdf,sdfY-sdf);let strokeGradientLength=max(length(strokeGradient),MIN_GRAD);let gradient=strokeGradient/vec2f(strokeGradientLength);let verticalGradient=abs(gradient.y);let horizontalDelta=mix(distance*1.1,distance*0.6,verticalGradient);let resultDelta=mix(distance,horizontalDelta,Font.hint);var alpha=smoothstep(0.5-resultDelta,resultDelta+0.5,sdf);alpha=pow(alpha,Font.hint*verticalGradient*0.2+1);if(alpha<THRESHOLD){discard;}let triplet=Font.subpx*gradient.x*0.5;let z=TRIPLET_FACTOR*triplet;let top=abs(z);let max=vec3f(-z,0,z);let average=vec3f(mix(top,-top-1,alpha));return clamp(max-average,vec3f(0),vec3f(1));}@fragment fn textFragment(input: TextVertexOutput)->@location(0)vec4f {let coverage=GetSubpixelCoverage(input.inverseTextureSize,input.distanceDelta,input.fontUV);return vec4f(mix(Font.back.rgb,Font.color.rgb,coverage),Font.color.a);}",$d="struct Shape{color: vec4f,matrix: mat3x3f};@group(0)@binding(0)var<uniform>resolution: vec3f;fn GetClipSpace(position: vec2f)->vec2f{let clipSpace=position/resolution.xy*2-1;return clipSpace*vec2f(1,-1);}@group(0)@binding(1)var<uniform>shape: Shape;fn GetVertexClipSpace(position: vec2f)->vec4f{let matrixPosition=shape.matrix*vec3f(position,1);let clipSpace=GetClipSpace(matrixPosition.xy);return vec4f(clipSpace,0,1);}@vertex fn shapeVertex(@location(0)position: vec2f)->@builtin(position)vec4f {return GetVertexClipSpace(position);}",Dd="@fragment fn shapeFragment()->@location(0)vec4f {return shape.color;}";const Od=Object.freeze(Object.defineProperty({__proto__:null,Empty:Th,Mipmaps:Ah,Quad:"const QUAD=array(vec2f(-1.0,-1.0),vec2f(1.0,-1.0),vec2f(1.0,1.0),vec2f(1.0,1.0),vec2f(-1.0,1.0),vec2f(-1.0,-1.0));fn GetQuadCoord(index: u32)->vec2f{return QUAD[index];}",Resolution:"@group(0)@binding(0)var<uniform>resolution: vec3f;fn GetClipSpace(position: vec2f)->vec2f{let clipSpace=position/resolution.xy*2-1;return clipSpace*vec2f(1,-1);}",SDFText:Gy,SDFTextDSB:Wy,Shape:`${$d}

${Dd}`,ShapeFragment:Dd,ShapeVertex:$d},Symbol.toStringTag,{value:"Module"}));var Rn,Ln,we,ar,Vs,it,_s,Wr,rc,ic;class Md{constructor(e,t){ee(this,it),ee(this,Rn),ee(this,Ln),ee(this,we),ee(this,ar),ee(this,Vs),!e&&ue(ie.DEVICE_NOT_REQUESTED),j(this,we,t),j(this,Rn,e)}CreateSampler(e){if(!e)return T(this,Rn).createSampler();const{addressModeUV:t,addressMode:n,minMagFilter:r,filter:i}=e;return t&&(e.addressModeU=e.addressModeV=t),n&&(e.addressModeU=e.addressModeV=e.addressModeW=n),r&&(e.minFilter=e.magFilter=r),i&&(e.minFilter=e.magFilter=e.mipmapFilter=i),T(this,Rn).createSampler(e)}CreateTexture(e){const t=e.label??"Texture",{format:n="rgba8unorm",usage:r=To.RENDER}=e;return T(this,Rn).createTexture({label:t,format:n,usage:r,...e})}WriteTexture(e,t){const{texture:n,mipLevel:r,origin:i,aspect:o,offset:a,rowsPerImage:l}=t,[u,c]=se(this,it,_s).call(this,n);let{bytesPerRow:h}=t;h??(h=(t.width??u)*Float32Array.BYTES_PER_ELEMENT),T(this,Rn).queue.writeTexture({texture:n,mipLevel:r,origin:i,aspect:o},e,{offset:a,bytesPerRow:h,rowsPerImage:l},se(this,it,Wr).call(this,{width:u,height:c,...t},"WriteTexture"))}CreateStorageTexture(e){let{size:t}=e;const n=To.STORAGE|e.usage,r=e.label??"Storage Texture",{format:i=this.PreferredStorageFormat}=e;return t=T(this,we)&&!t?T(this,we).CanvasSize:t,this.CreateTexture({label:r,size:t,format:i,...e,usage:n})}CreateBitmapImage(e,t){return createImageBitmap(e,t)}CreateTextureFromSource(e,t={}){const n=(t=typeof t=="boolean"&&{}||t).size,r=t.size,i=t.mipLevelCount??((t.mipmaps??!0)&&this.GetMipmapLevels(e)||void 0),o=Array.isArray(t.size)||!t.size?n??se(this,it,_s).call(this,e):[r.width,r.height];return this.CreateTexture({size:o,mipLevelCount:i,...t})}ImportExternalTexture(e,t,n){return T(this,Rn).importExternalTexture({source:e,label:t,colorSpace:n})}CreateMultisampleTexture(e=!1,t=4,n){var r;!T(this,we)&&ue(ie.LEGACY_RENDER_PIPELINE_NOT_FOUND,"creating a multisample texture.");const{width:i,height:o,format:a}=T(this,we).CurrentTexture;return!e&&T(this,Ln)&&T(this,Ln).width===i&&T(this,Ln).height===o||((r=T(this,Ln))==null||r.destroy(),j(this,Ln,this.CreateTexture({usage:GPUTextureUsage.RENDER_ATTACHMENT,label:n??"Multisample Texture",size:[i,o],sampleCount:t,format:a}))),T(this,Ln)}CopyImageToTexture(e,t={create:!0}){var n;let{create:r,texture:i}=t;const[o,a]=se(this,it,_s).call(this,e),{flipY:l,mipLevel:u,aspect:c,colorSpace:h,premultipliedAlpha:d,mipmaps:w}=t;return w===!1&&((n=r=typeof r=="object"&&r||{}).mipmaps??(n.mipmaps=!1)),!i&&!r&&ue(ie.TEXTURE_NOT_FOUND,"`CopyImageToTexture`."),i??(i=this.CreateTextureFromSource(e,r)),T(this,Rn).queue.copyExternalImageToTexture({source:e,origin:t.sourceOrigin,flipY:l},{texture:i,mipLevel:u,origin:t.destinationOrigin,aspect:c,colorSpace:h,premultipliedAlpha:d},se(this,it,Wr).call(this,{width:o,height:a,...t},"CopyImageToTexture")),(w??1)&&1<i.mipLevelCount&&(i.depthOrArrayLayers===1?this.GenerateMipmaps(i):this.GenerateCubeMipmaps(i)),i}CopyTextureToTexture(e){const{source:t,create:n}=e;let{srcTexture:r,dstTexture:i}=e;!T(this,we)&&ue(ie.LEGACY_RENDER_PIPELINE_NOT_FOUND,"copying a texture to a texture."),!r&&!t&&!n&&ue(ie.TEXTURE_NOT_FOUND,"`CopyTextureToTexture`."),r??(r=this.CreateTextureFromSource(t,n)),i??(i=this.CreateTextureFromSource(r,n));const{srcMipLevel:o,srcOrigin:a,srcAspect:l}=e,{dstMipLevel:u,dstOrigin:c,dstAspect:h}=e,[d,w]=se(this,it,_s).call(this,r);T(this,we).GetCommandEncoder(!0).copyTextureToTexture({texture:r,mipLevel:o,origin:a,aspect:l},{texture:i,mipLevel:u,origin:c,aspect:h},se(this,it,Wr).call(this,{width:d,height:w,...e},"CopyTextureToTexture"))}CopyTextureToBuffer(e){const{source:t,create:n}=e;let{texture:r,bytesPerRow:i}=e;!T(this,we)&&ue(ie.LEGACY_RENDER_PIPELINE_NOT_FOUND,"copying a texture to a buffer."),!r&&!t&&!n&&ue(ie.TEXTURE_NOT_FOUND,"`CopyTextureToBuffer`."),r??(r=this.CreateTextureFromSource(t,n));const[o,a]=se(this,it,_s).call(this,r),{buffer:l,offset:u,rowsPerImage:c,mipLevel:h,origin:d,aspect:w}=e;i??(i=(e.width??o)*Float32Array.BYTES_PER_ELEMENT),se(this,it,rc).call(this,i,"CopyTextureToBuffer"),T(this,we).GetCommandEncoder(!0).copyTextureToBuffer({texture:r,mipLevel:h,origin:d,aspect:w},{buffer:l,offset:u,bytesPerRow:i,rowsPerImage:c},se(this,it,Wr).call(this,{width:o,height:a,...e},"CopyTextureToBuffer"))}CopyBufferToTexture(e){const{source:t,create:n}=e;let{texture:r,bytesPerRow:i}=e;!T(this,we)&&ue(ie.LEGACY_RENDER_PIPELINE_NOT_FOUND,"copying a buffer to a texture."),!r&&!t&&!n&&ue(ie.TEXTURE_NOT_FOUND,"`CopyBufferToTexture`."),r??(r=this.CreateTextureFromSource(t,n));const[o,a]=se(this,it,_s).call(this,r),{buffer:l,offset:u,rowsPerImage:c,mipLevel:h,origin:d,aspect:w}=e;i??(i=(e.width??o)*Float32Array.BYTES_PER_ELEMENT),se(this,it,rc).call(this,i,"CopyBufferToTexture"),T(this,we).GetCommandEncoder(!0).copyBufferToTexture({buffer:l,offset:u,bytesPerRow:i,rowsPerImage:c},{texture:r,mipLevel:h,origin:d,aspect:w},se(this,it,Wr).call(this,{width:o,height:a,...e},"CopyBufferToTexture"))}get PreferredStorageFormat(){const e=En.PreferredCanvasFormat;return T(this,Rn).features.has("bgra8unorm-storage")&&e==="bgra8unorm"?e:"rgba8unorm"}GenerateCubeMipmaps(e){se(this,it,ic).call(this,e,{minMagFilter:Ao.LINEAR},t=>{for(let n=0;n<e.depthOrArrayLayers;++n)T(this,we).SetBindGroups(T(this,we).CreateBindGroup(T(this,we).CreateBindGroupEntries([T(this,ar),e.createView({baseMipLevel:t-1,arrayLayerCount:1,baseArrayLayer:n,mipLevelCount:1,dimension:"2d"})]))),T(this,we).CreatePassDescriptor(T(this,we).CreateColorAttachment(e.createView({arrayLayerCount:1,baseArrayLayer:n,mipLevelCount:1,dimension:"2d",baseMipLevel:t}))),T(this,we).Render(6,!1),T(this,we).DestroyCurrentPass()})}GenerateMipmaps(e){se(this,it,ic).call(this,e,{minFilter:Ao.LINEAR},t=>{T(this,we).SetBindGroups(T(this,we).CreateBindGroup(T(this,we).CreateBindGroupEntries([T(this,ar),e.createView({baseMipLevel:t-1,mipLevelCount:1})]))),T(this,we).CreatePassDescriptor(T(this,we).CreateColorAttachment(e.createView({baseMipLevel:t,mipLevelCount:1}))),T(this,we).Render(6,!1),T(this,we).DestroyCurrentPass()})}GetMipmapLevels(e){const[t,n]=se(this,it,_s).call(this,e);return 1+(0|Math.log2(Math.max(t,n)))}set LegacyRenderer(e){j(this,we,e)}SetRenderer(e){this.LegacyRenderer=e}Destroy(){var e;j(this,Ln,(e=T(this,Ln))==null?void 0:e.destroy())}}Rn=new WeakMap,Ln=new WeakMap,we=new WeakMap,ar=new WeakMap,Vs=new WeakMap,it=new WeakSet,_s=function(s){return s instanceof HTMLVideoElement?[s.videoWidth,s.videoHeight]:s instanceof VideoFrame?[s.codedWidth,s.codedHeight]:[s.width,s.height]},Wr=function(s,e){const{size:t,width:n,height:r,depthOrArrayLayers:i}=s;return!t&&!n&&ue(ie.TEXTURE_SIZE_NOT_FOUND,`\`${e}\` method.`),t??{width:n,height:r,depthOrArrayLayers:i}},rc=function(s,e){const t=s/256;t!==(0|t)&&Bt(ie.INVALID_BYTES_PER_ROW,`\`${e}\` options.`)},ic=function(s,e,t){!T(this,we)&&ue(ie.LEGACY_RENDER_PIPELINE_NOT_FOUND,"creating a texture with mipmaps."),T(this,we).SavePipelineState(),T(this,we).ResetPipelineState(),T(this,Vs)&&T(this,ar)||(j(this,Vs,T(this,we).CreateShaderModule(Ah)),j(this,ar,this.CreateSampler(e))),T(this,we).CreatePipeline({vertex:T(this,we).CreateVertexState(T(this,Vs)),fragment:T(this,we).CreateFragmentState(T(this,Vs),void 0,T(this,we).CreateTargetState(s.format))});for(let n=1;n<s.mipLevelCount;++n)t(n);T(this,we).SubmitCommandBuffer(),T(this,we).SetCommandEncoder(void 0),T(this,we).RestorePipelineState(),j(this,Vs,j(this,ar,void 0))};var Bn,Fn,Me,lr,qs,ot,vs,Gr,oc,ac;class Ch{constructor(e,t){ee(this,ot),ee(this,Bn),ee(this,Fn),ee(this,Me),ee(this,lr),ee(this,qs),!e&&ue(ie.DEVICE_NOT_REQUESTED),j(this,Me,t),j(this,Bn,e)}CreateSampler(e){if(!e)return T(this,Bn).createSampler();const{addressModeUV:t,addressMode:n,minMagFilter:r,filter:i}=e;return t&&(e.addressModeU=e.addressModeV=t),n&&(e.addressModeU=e.addressModeV=e.addressModeW=n),r&&(e.minFilter=e.magFilter=r),i&&(e.minFilter=e.magFilter=e.mipmapFilter=i),T(this,Bn).createSampler(e)}CreateTexture(e){const t=e.label??"Texture",{format:n=En.PreferredCanvasFormat,usage:r=To.RENDER}=e;return T(this,Bn).createTexture({label:t,format:n,usage:r,...e})}WriteTexture(e,t){const{texture:n,mipLevel:r,origin:i,aspect:o,offset:a,rowsPerImage:l}=t,[u,c]=se(this,ot,vs).call(this,n);let{bytesPerRow:h}=t;h??(h=(t.width??u)*Float32Array.BYTES_PER_ELEMENT),T(this,Bn).queue.writeTexture({texture:n,mipLevel:r,origin:i,aspect:o},e,{offset:a,bytesPerRow:h,rowsPerImage:l},se(this,ot,Gr).call(this,{width:u,height:c,...t},"WriteTexture"))}CreateStorageTexture(e){let{size:t}=e;const n=To.STORAGE|e.usage,r=e.label??"Storage Texture",{format:i=this.PreferredStorageFormat}=e;return t=T(this,Me)&&!t?T(this,Me).CanvasSize:t,this.CreateTexture({label:r,size:t,format:i,...e,usage:n})}CreateBitmapImage(e,t){return createImageBitmap(e,t)}CreateTextureFromSource(e,t={}){const n=(t=typeof t=="boolean"&&{}||t).size,r=t.size,i=t.mipLevelCount??((t.mipmaps??!0)&&this.GetMipmapLevels(e)||void 0),o=Array.isArray(t.size)||!t.size?n??se(this,ot,vs).call(this,e):[r.width,r.height];return this.CreateTexture({size:o,mipLevelCount:i,...t})}ImportExternalTexture(e,t,n){return T(this,Bn).importExternalTexture({source:e,label:t,colorSpace:n})}async LoadExternalImageSource(e,t={}){return new Promise(n=>{const r=new Image;for(const i in t)r[i]=t[i];r.onload=()=>n(r),r.src=e})}CreateMultisampleTexture(e=!1,t=4,n){var r;!T(this,Me)&&ue(ie.RENDERER_NOT_FOUND,"creating a multisample texture.");const{width:i,height:o,format:a}=T(this,Me).CurrentTexture;return!e&&T(this,Fn)&&T(this,Fn).width===i&&T(this,Fn).height===o||((r=T(this,Fn))==null||r.destroy(),j(this,Fn,this.CreateTexture({usage:GPUTextureUsage.RENDER_ATTACHMENT,label:n??"Multisample Texture",size:[i,o],sampleCount:t,format:a}))),T(this,Fn)}async CopyImageToTexture(e,t={create:!0}){var n;let{create:r,texture:i}=t;const[o,a]=se(this,ot,vs).call(this,e),{flipY:l,mipLevel:u,aspect:c,colorSpace:h,premultipliedAlpha:d,mipmaps:w}=t;return w===!1&&((n=r=typeof r=="object"&&r||{}).mipmaps??(n.mipmaps=!1)),!i&&!r&&ue(ie.TEXTURE_NOT_FOUND,"`CopyImageToTexture`."),i??(i=this.CreateTextureFromSource(e,r)),T(this,Bn).queue.copyExternalImageToTexture({source:e,origin:t.sourceOrigin,flipY:l},{texture:i,mipLevel:u,origin:t.destinationOrigin,aspect:c,colorSpace:h,premultipliedAlpha:d},se(this,ot,Gr).call(this,{width:o,height:a,...t},"CopyImageToTexture")),(w??1)&&1<i.mipLevelCount&&(i.depthOrArrayLayers===1?await this.GenerateMipmaps(i):await this.GenerateCubeMipmaps(i)),i}async GenerateCubeMipmaps(e){return se(this,ot,ac).call(this,e,{minMagFilter:Ao.LINEAR},(t,n)=>{for(let r=0;r<e.depthOrArrayLayers;++r)t.SetBindGroups(t.CreateBindGroup(t.CreateBindGroupEntries([T(this,lr),e.createView({baseMipLevel:n-1,arrayLayerCount:1,baseArrayLayer:r,mipLevelCount:1,dimension:"2d"})]))),T(this,Me).CreatePassDescriptor(T(this,Me).CreateColorAttachment(void 0,e.createView({arrayLayerCount:1,baseArrayLayer:r,mipLevelCount:1,dimension:"2d",baseMipLevel:n}))),T(this,Me).Render(!1)})}async GenerateMipmaps(e){return se(this,ot,ac).call(this,e,{minFilter:Ao.LINEAR},(t,n)=>{t.SetBindGroups(t.CreateBindGroup(t.CreateBindGroupEntries([T(this,lr),e.createView({baseMipLevel:n-1,mipLevelCount:1})]))),T(this,Me).CreatePassDescriptor(T(this,Me).CreateColorAttachment(void 0,e.createView({baseMipLevel:n,mipLevelCount:1}))),T(this,Me).Render(!1)})}CopyTextureToTexture(e){const{source:t,create:n}=e;let{srcTexture:r,dstTexture:i}=e;!T(this,Me)&&ue(ie.RENDERER_NOT_FOUND,"copying a texture to a texture."),!r&&!t&&!n&&ue(ie.TEXTURE_NOT_FOUND,"`CopyTextureToTexture`."),r??(r=this.CreateTextureFromSource(t,n)),i??(i=this.CreateTextureFromSource(r,n));const{srcMipLevel:o,srcOrigin:a,srcAspect:l}=e,{dstMipLevel:u,dstOrigin:c,dstAspect:h}=e,[d,w]=se(this,ot,vs).call(this,r);T(this,Me).GetCommandEncoder(!0).copyTextureToTexture({texture:r,mipLevel:o,origin:a,aspect:l},{texture:i,mipLevel:u,origin:c,aspect:h},se(this,ot,Gr).call(this,{width:d,height:w,...e},"CopyTextureToTexture"))}CopyTextureToBuffer(e){const{source:t,create:n}=e;let{texture:r,bytesPerRow:i}=e;!T(this,Me)&&ue(ie.RENDERER_NOT_FOUND,"copying a texture to a buffer."),!r&&!t&&!n&&ue(ie.TEXTURE_NOT_FOUND,"`CopyTextureToBuffer`."),r??(r=this.CreateTextureFromSource(t,n));const[o,a]=se(this,ot,vs).call(this,r),{buffer:l,offset:u,rowsPerImage:c,mipLevel:h,origin:d,aspect:w}=e;i??(i=(e.width??o)*Float32Array.BYTES_PER_ELEMENT),se(this,ot,oc).call(this,i,"CopyTextureToBuffer"),T(this,Me).GetCommandEncoder(!0).copyTextureToBuffer({texture:r,mipLevel:h,origin:d,aspect:w},{buffer:l,offset:u,bytesPerRow:i,rowsPerImage:c},se(this,ot,Gr).call(this,{width:o,height:a,...e},"CopyTextureToBuffer"))}CopyBufferToTexture(e){const{source:t,create:n}=e;let{texture:r,bytesPerRow:i}=e;!T(this,Me)&&ue(ie.RENDERER_NOT_FOUND,"copying a buffer to a texture."),!r&&!t&&!n&&ue(ie.TEXTURE_NOT_FOUND,"`CopyBufferToTexture`."),r??(r=this.CreateTextureFromSource(t,n));const[o,a]=se(this,ot,vs).call(this,r),{buffer:l,offset:u,rowsPerImage:c,mipLevel:h,origin:d,aspect:w}=e;i??(i=(e.width??o)*Float32Array.BYTES_PER_ELEMENT),se(this,ot,oc).call(this,i,"CopyBufferToTexture"),T(this,Me).GetCommandEncoder(!0).copyBufferToTexture({buffer:l,offset:u,bytesPerRow:i,rowsPerImage:c},{texture:r,mipLevel:h,origin:d,aspect:w},se(this,ot,Gr).call(this,{width:o,height:a,...e},"CopyBufferToTexture"))}get PreferredStorageFormat(){const e=En.PreferredCanvasFormat;return T(this,Bn).features.has("bgra8unorm-storage")&&e==="bgra8unorm"?e:"rgba8unorm"}GetMipmapLevels(e){const[t,n]=se(this,ot,vs).call(this,e);return 1+(0|Math.log2(Math.max(t,n)))}set Renderer(e){j(this,Me,e)}Destroy(){var e;j(this,Fn,(e=T(this,Fn))==null?void 0:e.destroy())}}Bn=new WeakMap,Fn=new WeakMap,Me=new WeakMap,lr=new WeakMap,qs=new WeakMap,ot=new WeakSet,vs=function(s){return s instanceof HTMLVideoElement?[s.videoWidth,s.videoHeight]:s instanceof VideoFrame?[s.codedWidth,s.codedHeight]:[s.width,s.height]},Gr=function(s,e){const{size:t,width:n,height:r,depthOrArrayLayers:i}=s;return!t&&!n&&ue(ie.TEXTURE_SIZE_NOT_FOUND,`\`${e}\` method.`),t??{width:n,height:r,depthOrArrayLayers:i}},oc=function(s,e){const t=s/256;t!==(0|t)&&Bt(ie.INVALID_BYTES_PER_ROW,`\`${e}\` options.`)},ac=async function(s,e,t){!T(this,Me)&&ue(ie.RENDERER_NOT_FOUND,"creating a texture with mipmaps.");const n=new(T(this,Me)).Pipeline;n.DestroyPassEncoder=!0,n.SetDrawParams(6),T(this,qs)&&T(this,lr)||(j(this,qs,n.CreateShaderModule(Ah)),j(this,lr,this.CreateSampler(e))),await T(this,Me).AddPipeline(n,{vertex:n.CreateVertexState(T(this,qs)),fragment:n.CreateFragmentState(T(this,qs),void 0,n.CreateColorTargetState(s.format))});for(let r=1;r<s.mipLevelCount;++r)t(n,r);T(this,Me).SubmitCommandBuffer(),T(this,Me).CommandEncoder=void 0,T(this,Me).RemovePipeline(n),j(this,qs,j(this,lr,void 0))};const JA=(Pd=Array,Rd=s=>s.fill(0),class extends Pd{constructor(...s){super(...s),Rd(this)}});var Pd,Rd;let be=1e-6;const Ld=new Map;function Vy(s){let e=Ld.get(s);return e||(e=(t=>{function n(p=0,y=0){const x=new t(2);return p!==void 0&&(x[0]=p,y!==void 0&&(x[1]=y)),x}function r(p,y,x){const I=x??new t(2);return I[0]=p[0]-y[0],I[1]=p[1]-y[1],I}const i=r;function o(p,y,x,I){const N=I??new t(2);return N[0]=p[0]+x*(y[0]-p[0]),N[1]=p[1]+x*(y[1]-p[1]),N}function a(p,y,x){const I=x??new t(2);return I[0]=p[0]*y,I[1]=p[1]*y,I}const l=a;function u(p,y){const x=y??new t(2);return x[0]=1/p[0],x[1]=1/p[1],x}const c=u;function h(p,y){return p[0]*y[0]+p[1]*y[1]}function d(p){const y=p[0],x=p[1];return Math.sqrt(y*y+x*x)}const w=d;function k(p){const y=p[0],x=p[1];return y*y+x*x}const A=k;function m(p,y){const x=p[0]-y[0],I=p[1]-y[1];return Math.sqrt(x*x+I*I)}const S=m;function b(p,y){const x=p[0]-y[0],I=p[1]-y[1];return x*x+I*I}const f=b;function v(p,y){const x=y??new t(2),I=p[0],N=p[1],L=Math.sqrt(I*I+N*N);return L>1e-5?(x[0]=I/L,x[1]=N/L):(x[0]=0,x[1]=0),x}function _(p,y){const x=y??new t(2);return x[0]=p[0],x[1]=p[1],x}const E=_;function D(p,y,x){const I=x??new t(2);return I[0]=p[0]*y[0],I[1]=p[1]*y[1],I}const M=D;function $(p,y,x){const I=x??new t(2);return I[0]=p[0]/y[0],I[1]=p[1]/y[1],I}const C=$;function g(p,y,x){const I=x??new t(2);return v(p,I),a(I,y,I)}return{create:n,fromValues:n,set(p,y,x){const I=x??new t(2);return I[0]=p,I[1]=y,I},ceil(p,y){const x=y??new t(2);return x[0]=Math.ceil(p[0]),x[1]=Math.ceil(p[1]),x},floor(p,y){const x=y??new t(2);return x[0]=Math.floor(p[0]),x[1]=Math.floor(p[1]),x},round(p,y){const x=y??new t(2);return x[0]=Math.round(p[0]),x[1]=Math.round(p[1]),x},clamp(p,y=0,x=1,I){const N=I??new t(2);return N[0]=Math.min(x,Math.max(y,p[0])),N[1]=Math.min(x,Math.max(y,p[1])),N},add(p,y,x){const I=x??new t(2);return I[0]=p[0]+y[0],I[1]=p[1]+y[1],I},addScaled(p,y,x,I){const N=I??new t(2);return N[0]=p[0]+y[0]*x,N[1]=p[1]+y[1]*x,N},angle(p,y){const x=p[0],I=p[1],N=y[0],L=y[1],W=Math.sqrt(x*x+I*I)*Math.sqrt(N*N+L*L),X=W&&h(p,y)/W;return Math.acos(X)},subtract:r,sub:i,equalsApproximately(p,y){return Math.abs(p[0]-y[0])<be&&Math.abs(p[1]-y[1])<be},equals(p,y){return p[0]===y[0]&&p[1]===y[1]},lerp:o,lerpV(p,y,x,I){const N=I??new t(2);return N[0]=p[0]+x[0]*(y[0]-p[0]),N[1]=p[1]+x[1]*(y[1]-p[1]),N},max(p,y,x){const I=x??new t(2);return I[0]=Math.max(p[0],y[0]),I[1]=Math.max(p[1],y[1]),I},min(p,y,x){const I=x??new t(2);return I[0]=Math.min(p[0],y[0]),I[1]=Math.min(p[1],y[1]),I},mulScalar:a,scale:l,divScalar(p,y,x){const I=x??new t(2);return I[0]=p[0]/y,I[1]=p[1]/y,I},inverse:u,invert:c,cross(p,y,x){const I=x??new t(3),N=p[0]*y[1]-p[1]*y[0];return I[0]=0,I[1]=0,I[2]=N,I},dot:h,length:d,len:w,lengthSq:k,lenSq:A,distance:m,dist:S,distanceSq:b,distSq:f,normalize:v,negate(p,y){const x=y??new t(2);return x[0]=-p[0],x[1]=-p[1],x},copy:_,clone:E,multiply:D,mul:M,divide:$,div:C,random(p=1,y){const x=y??new t(2),I=2*Math.random()*Math.PI;return x[0]=Math.cos(I)*p,x[1]=Math.sin(I)*p,x},zero(p){const y=p??new t(2);return y[0]=0,y[1]=0,y},transformMat4(p,y,x){const I=x??new t(2),N=p[0],L=p[1];return I[0]=N*y[0]+L*y[4]+y[12],I[1]=N*y[1]+L*y[5]+y[13],I},transformMat3(p,y,x){const I=x??new t(2),N=p[0],L=p[1];return I[0]=y[0]*N+y[4]*L+y[8],I[1]=y[1]*N+y[5]*L+y[9],I},rotate(p,y,x,I){const N=I??new t(2),L=p[0]-y[0],W=p[1]-y[1],X=Math.sin(x),V=Math.cos(x);return N[0]=L*V-W*X+y[0],N[1]=L*X+W*V+y[1],N},setLength:g,truncate(p,y,x){const I=x??new t(2);return d(p)>y?g(p,y,I):_(p,I)},midpoint(p,y,x){return o(p,y,.5,x??new t(2))}}})(s),Ld.set(s,e)),e}const Bd=new Map;function Hl(s){let e=Bd.get(s);return e||(e=(t=>{function n(p,y,x){const I=new t(3);return p!==void 0&&(I[0]=p,y!==void 0&&(I[1]=y,x!==void 0&&(I[2]=x))),I}function r(p,y,x){const I=x??new t(3);return I[0]=p[0]-y[0],I[1]=p[1]-y[1],I[2]=p[2]-y[2],I}const i=r;function o(p,y,x,I){const N=I??new t(3);return N[0]=p[0]+x*(y[0]-p[0]),N[1]=p[1]+x*(y[1]-p[1]),N[2]=p[2]+x*(y[2]-p[2]),N}function a(p,y,x){const I=x??new t(3);return I[0]=p[0]*y,I[1]=p[1]*y,I[2]=p[2]*y,I}const l=a;function u(p,y){const x=y??new t(3);return x[0]=1/p[0],x[1]=1/p[1],x[2]=1/p[2],x}const c=u;function h(p,y){return p[0]*y[0]+p[1]*y[1]+p[2]*y[2]}function d(p){const y=p[0],x=p[1],I=p[2];return Math.sqrt(y*y+x*x+I*I)}const w=d;function k(p){const y=p[0],x=p[1],I=p[2];return y*y+x*x+I*I}const A=k;function m(p,y){const x=p[0]-y[0],I=p[1]-y[1],N=p[2]-y[2];return Math.sqrt(x*x+I*I+N*N)}const S=m;function b(p,y){const x=p[0]-y[0],I=p[1]-y[1],N=p[2]-y[2];return x*x+I*I+N*N}const f=b;function v(p,y){const x=y??new t(3),I=p[0],N=p[1],L=p[2],W=Math.sqrt(I*I+N*N+L*L);return W>1e-5?(x[0]=I/W,x[1]=N/W,x[2]=L/W):(x[0]=0,x[1]=0,x[2]=0),x}function _(p,y){const x=y??new t(3);return x[0]=p[0],x[1]=p[1],x[2]=p[2],x}const E=_;function D(p,y,x){const I=x??new t(3);return I[0]=p[0]*y[0],I[1]=p[1]*y[1],I[2]=p[2]*y[2],I}const M=D;function $(p,y,x){const I=x??new t(3);return I[0]=p[0]/y[0],I[1]=p[1]/y[1],I[2]=p[2]/y[2],I}const C=$;function g(p,y,x){const I=x??new t(3);return v(p,I),a(I,y,I)}return{create:n,fromValues:n,set(p,y,x,I){const N=I??new t(3);return N[0]=p,N[1]=y,N[2]=x,N},ceil(p,y){const x=y??new t(3);return x[0]=Math.ceil(p[0]),x[1]=Math.ceil(p[1]),x[2]=Math.ceil(p[2]),x},floor(p,y){const x=y??new t(3);return x[0]=Math.floor(p[0]),x[1]=Math.floor(p[1]),x[2]=Math.floor(p[2]),x},round(p,y){const x=y??new t(3);return x[0]=Math.round(p[0]),x[1]=Math.round(p[1]),x[2]=Math.round(p[2]),x},clamp(p,y=0,x=1,I){const N=I??new t(3);return N[0]=Math.min(x,Math.max(y,p[0])),N[1]=Math.min(x,Math.max(y,p[1])),N[2]=Math.min(x,Math.max(y,p[2])),N},add(p,y,x){const I=x??new t(3);return I[0]=p[0]+y[0],I[1]=p[1]+y[1],I[2]=p[2]+y[2],I},addScaled(p,y,x,I){const N=I??new t(3);return N[0]=p[0]+y[0]*x,N[1]=p[1]+y[1]*x,N[2]=p[2]+y[2]*x,N},angle(p,y){const x=p[0],I=p[1],N=p[2],L=y[0],W=y[1],X=y[2],V=Math.sqrt(x*x+I*I+N*N)*Math.sqrt(L*L+W*W+X*X),Z=V&&h(p,y)/V;return Math.acos(Z)},subtract:r,sub:i,equalsApproximately(p,y){return Math.abs(p[0]-y[0])<be&&Math.abs(p[1]-y[1])<be&&Math.abs(p[2]-y[2])<be},equals(p,y){return p[0]===y[0]&&p[1]===y[1]&&p[2]===y[2]},lerp:o,lerpV(p,y,x,I){const N=I??new t(3);return N[0]=p[0]+x[0]*(y[0]-p[0]),N[1]=p[1]+x[1]*(y[1]-p[1]),N[2]=p[2]+x[2]*(y[2]-p[2]),N},max(p,y,x){const I=x??new t(3);return I[0]=Math.max(p[0],y[0]),I[1]=Math.max(p[1],y[1]),I[2]=Math.max(p[2],y[2]),I},min(p,y,x){const I=x??new t(3);return I[0]=Math.min(p[0],y[0]),I[1]=Math.min(p[1],y[1]),I[2]=Math.min(p[2],y[2]),I},mulScalar:a,scale:l,divScalar(p,y,x){const I=x??new t(3);return I[0]=p[0]/y,I[1]=p[1]/y,I[2]=p[2]/y,I},inverse:u,invert:c,cross(p,y,x){const I=x??new t(3),N=p[2]*y[0]-p[0]*y[2],L=p[0]*y[1]-p[1]*y[0];return I[0]=p[1]*y[2]-p[2]*y[1],I[1]=N,I[2]=L,I},dot:h,length:d,len:w,lengthSq:k,lenSq:A,distance:m,dist:S,distanceSq:b,distSq:f,normalize:v,negate(p,y){const x=y??new t(3);return x[0]=-p[0],x[1]=-p[1],x[2]=-p[2],x},copy:_,clone:E,multiply:D,mul:M,divide:$,div:C,random(p=1,y){const x=y??new t(3),I=2*Math.random()*Math.PI,N=2*Math.random()-1,L=Math.sqrt(1-N*N)*p;return x[0]=Math.cos(I)*L,x[1]=Math.sin(I)*L,x[2]=N*p,x},zero(p){const y=p??new t(3);return y[0]=0,y[1]=0,y[2]=0,y},transformMat4(p,y,x){const I=x??new t(3),N=p[0],L=p[1],W=p[2],X=y[3]*N+y[7]*L+y[11]*W+y[15]||1;return I[0]=(y[0]*N+y[4]*L+y[8]*W+y[12])/X,I[1]=(y[1]*N+y[5]*L+y[9]*W+y[13])/X,I[2]=(y[2]*N+y[6]*L+y[10]*W+y[14])/X,I},transformMat4Upper3x3(p,y,x){const I=x??new t(3),N=p[0],L=p[1],W=p[2];return I[0]=N*y[0]+L*y[4]+W*y[8],I[1]=N*y[1]+L*y[5]+W*y[9],I[2]=N*y[2]+L*y[6]+W*y[10],I},transformMat3(p,y,x){const I=x??new t(3),N=p[0],L=p[1],W=p[2];return I[0]=N*y[0]+L*y[4]+W*y[8],I[1]=N*y[1]+L*y[5]+W*y[9],I[2]=N*y[2]+L*y[6]+W*y[10],I},transformQuat(p,y,x){const I=x??new t(3),N=y[0],L=y[1],W=y[2],X=2*y[3],V=p[0],Z=p[1],te=p[2],oe=L*te-W*Z,ce=W*V-N*te,fe=N*Z-L*V;return I[0]=V+oe*X+2*(L*fe-W*ce),I[1]=Z+ce*X+2*(W*oe-N*fe),I[2]=te+fe*X+2*(N*ce-L*oe),I},getTranslation(p,y){const x=y??new t(3);return x[0]=p[12],x[1]=p[13],x[2]=p[14],x},getAxis(p,y,x){const I=x??new t(3),N=4*y;return I[0]=p[N+0],I[1]=p[N+1],I[2]=p[N+2],I},getScaling(p,y){const x=y??new t(3),I=p[0],N=p[1],L=p[2],W=p[4],X=p[5],V=p[6],Z=p[8],te=p[9],oe=p[10];return x[0]=Math.sqrt(I*I+N*N+L*L),x[1]=Math.sqrt(W*W+X*X+V*V),x[2]=Math.sqrt(Z*Z+te*te+oe*oe),x},rotateX(p,y,x,I){const N=I??new t(3),L=[],W=[];return L[0]=p[0]-y[0],L[1]=p[1]-y[1],L[2]=p[2]-y[2],W[0]=L[0],W[1]=L[1]*Math.cos(x)-L[2]*Math.sin(x),W[2]=L[1]*Math.sin(x)+L[2]*Math.cos(x),N[0]=W[0]+y[0],N[1]=W[1]+y[1],N[2]=W[2]+y[2],N},rotateY(p,y,x,I){const N=I??new t(3),L=[],W=[];return L[0]=p[0]-y[0],L[1]=p[1]-y[1],L[2]=p[2]-y[2],W[0]=L[2]*Math.sin(x)+L[0]*Math.cos(x),W[1]=L[1],W[2]=L[2]*Math.cos(x)-L[0]*Math.sin(x),N[0]=W[0]+y[0],N[1]=W[1]+y[1],N[2]=W[2]+y[2],N},rotateZ(p,y,x,I){const N=I??new t(3),L=[],W=[];return L[0]=p[0]-y[0],L[1]=p[1]-y[1],L[2]=p[2]-y[2],W[0]=L[0]*Math.cos(x)-L[1]*Math.sin(x),W[1]=L[0]*Math.sin(x)+L[1]*Math.cos(x),W[2]=L[2],N[0]=W[0]+y[0],N[1]=W[1]+y[1],N[2]=W[2]+y[2],N},setLength:g,truncate(p,y,x){const I=x??new t(3);return d(p)>y?g(p,y,I):_(p,I)},midpoint(p,y,x){return o(p,y,.5,x??new t(3))}}})(s),Bd.set(s,e)),e}const Fd=new Map;function eC(s){let e=Fd.get(s);return e||(e=(t=>{const n=Vy(t),r=Hl(t);function i(m,S,b){const f=b??new t(12);return f[0]=m[0]*S,f[1]=m[1]*S,f[2]=m[2]*S,f[4]=m[4]*S,f[5]=m[5]*S,f[6]=m[6]*S,f[8]=m[8]*S,f[9]=m[9]*S,f[10]=m[10]*S,f}const o=i;function a(m,S){const b=S??new t(12);return b[0]=m[0],b[1]=m[1],b[2]=m[2],b[4]=m[4],b[5]=m[5],b[6]=m[6],b[8]=m[8],b[9]=m[9],b[10]=m[10],b}const l=a;function u(m){const S=m??new t(12);return S[0]=1,S[1]=0,S[2]=0,S[4]=0,S[5]=1,S[6]=0,S[8]=0,S[9]=0,S[10]=1,S}function c(m,S){const b=S??new t(12),f=m[0],v=m[1],_=m[2],E=m[4],D=m[5],M=m[6],$=m[8],C=m[9],g=m[10],p=g*D-M*C,y=-g*E+M*$,x=C*E-D*$,I=1/(f*p+v*y+_*x);return b[0]=p*I,b[1]=(-g*v+_*C)*I,b[2]=(M*v-_*D)*I,b[4]=y*I,b[5]=(g*f-_*$)*I,b[6]=(-M*f+_*E)*I,b[8]=x*I,b[9]=(-C*f+v*$)*I,b[10]=(D*f-v*E)*I,b}const h=c;function d(m,S,b){const f=b??new t(12),v=m[0],_=m[1],E=m[2],D=m[4],M=m[5],$=m[6],C=m[8],g=m[9],p=m[10],y=S[0],x=S[1],I=S[2],N=S[4],L=S[5],W=S[6],X=S[8],V=S[9],Z=S[10];return f[0]=v*y+D*x+C*I,f[1]=_*y+M*x+g*I,f[2]=E*y+$*x+p*I,f[4]=v*N+D*L+C*W,f[5]=_*N+M*L+g*W,f[6]=E*N+$*L+p*W,f[8]=v*X+D*V+C*Z,f[9]=_*X+M*V+g*Z,f[10]=E*X+$*V+p*Z,f}const w=d;function k(m,S){const b=S??new t(12),f=Math.cos(m),v=Math.sin(m);return b[0]=f,b[1]=v,b[2]=0,b[4]=-v,b[5]=f,b[6]=0,b[8]=0,b[9]=0,b[10]=1,b}function A(m,S,b){const f=b??new t(12),v=m[0],_=m[1],E=m[2],D=m[4],M=m[5],$=m[6],C=Math.cos(S),g=Math.sin(S);return f[0]=C*v+g*D,f[1]=C*_+g*M,f[2]=C*E+g*$,f[4]=C*D-g*v,f[5]=C*M-g*_,f[6]=C*$-g*E,m!==f&&(f[8]=m[8],f[9]=m[9],f[10]=m[10]),f}return{add(m,S,b){const f=b??new t(12);return f[0]=m[0]+S[0],f[1]=m[1]+S[1],f[2]=m[2]+S[2],f[4]=m[4]+S[4],f[5]=m[5]+S[5],f[6]=m[6]+S[6],f[8]=m[8]+S[8],f[9]=m[9]+S[9],f[10]=m[10]+S[10],f},clone:l,copy:a,create(m,S,b,f,v,_,E,D,M){const $=new t(12);return $[3]=0,$[7]=0,$[11]=0,m!==void 0&&($[0]=m,S!==void 0&&($[1]=S,b!==void 0&&($[2]=b,f!==void 0&&($[4]=f,v!==void 0&&($[5]=v,_!==void 0&&($[6]=_,E!==void 0&&($[8]=E,D!==void 0&&($[9]=D,M!==void 0&&($[10]=M))))))))),$},determinant(m){const S=m[0],b=m[1],f=m[2],v=m[4],_=m[5],E=m[6],D=m[8],M=m[9],$=m[10];return S*(_*$-M*E)-v*(b*$-M*f)+D*(b*E-_*f)},equals(m,S){return m[0]===S[0]&&m[1]===S[1]&&m[2]===S[2]&&m[4]===S[4]&&m[5]===S[5]&&m[6]===S[6]&&m[8]===S[8]&&m[9]===S[9]&&m[10]===S[10]},equalsApproximately(m,S){return Math.abs(m[0]-S[0])<be&&Math.abs(m[1]-S[1])<be&&Math.abs(m[2]-S[2])<be&&Math.abs(m[4]-S[4])<be&&Math.abs(m[5]-S[5])<be&&Math.abs(m[6]-S[6])<be&&Math.abs(m[8]-S[8])<be&&Math.abs(m[9]-S[9])<be&&Math.abs(m[10]-S[10])<be},fromMat4(m,S){const b=S??new t(12);return b[0]=m[0],b[1]=m[1],b[2]=m[2],b[3]=0,b[4]=m[4],b[5]=m[5],b[6]=m[6],b[7]=0,b[8]=m[8],b[9]=m[9],b[10]=m[10],b[11]=0,b},fromQuat(m,S){const b=S??new t(12),f=m[0],v=m[1],_=m[2],E=m[3],D=f+f,M=v+v,$=_+_,C=f*D,g=v*D,p=v*M,y=_*D,x=_*M,I=_*$,N=E*D,L=E*M,W=E*$;return b[0]=1-p-I,b[1]=g+W,b[2]=y-L,b[3]=0,b[4]=g-W,b[5]=1-C-I,b[6]=x+N,b[7]=0,b[8]=y+L,b[9]=x-N,b[10]=1-C-p,b[11]=0,b},get3DScaling(m,S){const b=S??r.create(),f=m[0],v=m[1],_=m[2],E=m[4],D=m[5],M=m[6],$=m[8],C=m[9],g=m[10];return b[0]=Math.sqrt(f*f+v*v+_*_),b[1]=Math.sqrt(E*E+D*D+M*M),b[2]=Math.sqrt($*$+C*C+g*g),b},getAxis(m,S,b){const f=b??n.create(),v=4*S;return f[0]=m[v+0],f[1]=m[v+1],f},getScaling(m,S){const b=S??n.create(),f=m[0],v=m[1],_=m[4],E=m[5];return b[0]=Math.sqrt(f*f+v*v),b[1]=Math.sqrt(_*_+E*E),b},getTranslation(m,S){const b=S??n.create();return b[0]=m[8],b[1]=m[9],b},identity:u,inverse:c,invert:h,mul:w,mulScalar:o,multiply:d,multiplyScalar:i,negate(m,S){const b=S??new t(12);return b[0]=-m[0],b[1]=-m[1],b[2]=-m[2],b[4]=-m[4],b[5]=-m[5],b[6]=-m[6],b[8]=-m[8],b[9]=-m[9],b[10]=-m[10],b},rotate:A,rotateX(m,S,b){const f=b??new t(12),v=m[4],_=m[5],E=m[6],D=m[8],M=m[9],$=m[10],C=Math.cos(S),g=Math.sin(S);return f[4]=C*v+g*D,f[5]=C*_+g*M,f[6]=C*E+g*$,f[8]=C*D-g*v,f[9]=C*M-g*_,f[10]=C*$-g*E,m!==f&&(f[0]=m[0],f[1]=m[1],f[2]=m[2]),f},rotateY(m,S,b){const f=b??new t(12),v=m[0],_=m[1],E=m[2],D=m[8],M=m[9],$=m[10],C=Math.cos(S),g=Math.sin(S);return f[0]=C*v-g*D,f[1]=C*_-g*M,f[2]=C*E-g*$,f[8]=C*D+g*v,f[9]=C*M+g*_,f[10]=C*$+g*E,m!==f&&(f[4]=m[4],f[5]=m[5],f[6]=m[6]),f},rotateZ:A,rotation:k,rotationX(m,S){const b=S??new t(12),f=Math.cos(m),v=Math.sin(m);return b[0]=1,b[1]=0,b[2]=0,b[4]=0,b[5]=f,b[6]=v,b[8]=0,b[9]=-v,b[10]=f,b},rotationY(m,S){const b=S??new t(12),f=Math.cos(m),v=Math.sin(m);return b[0]=f,b[1]=0,b[2]=-v,b[4]=0,b[5]=1,b[6]=0,b[8]=v,b[9]=0,b[10]=f,b},rotationZ:k,scale(m,S,b){const f=b??new t(12),v=S[0],_=S[1];return f[0]=v*m[0],f[1]=v*m[1],f[2]=v*m[2],f[4]=_*m[4],f[5]=_*m[5],f[6]=_*m[6],m!==f&&(f[8]=m[8],f[9]=m[9],f[10]=m[10]),f},scale3D(m,S,b){const f=b??new t(12),v=S[0],_=S[1],E=S[2];return f[0]=v*m[0],f[1]=v*m[1],f[2]=v*m[2],f[4]=_*m[4],f[5]=_*m[5],f[6]=_*m[6],f[8]=E*m[8],f[9]=E*m[9],f[10]=E*m[10],f},scaling(m,S){const b=S??new t(12);return b[0]=m[0],b[1]=0,b[2]=0,b[4]=0,b[5]=m[1],b[6]=0,b[8]=0,b[9]=0,b[10]=1,b},scaling3D(m,S){const b=S??new t(12);return b[0]=m[0],b[1]=0,b[2]=0,b[4]=0,b[5]=m[1],b[6]=0,b[8]=0,b[9]=0,b[10]=m[2],b},set(m,S,b,f,v,_,E,D,M,$){const C=$??new t(12);return C[0]=m,C[1]=S,C[2]=b,C[3]=0,C[4]=f,C[5]=v,C[6]=_,C[7]=0,C[8]=E,C[9]=D,C[10]=M,C[11]=0,C},setAxis(m,S,b,f){const v=f===m?m:a(m,f),_=4*b;return v[_+0]=S[0],v[_+1]=S[1],v},setTranslation(m,S,b){const f=b??u();return m!==f&&(f[0]=m[0],f[1]=m[1],f[2]=m[2],f[4]=m[4],f[5]=m[5],f[6]=m[6]),f[8]=S[0],f[9]=S[1],f[10]=1,f},translate(m,S,b){const f=b??new t(12),v=S[0],_=S[1],E=m[0],D=m[1],M=m[2],$=m[4],C=m[5],g=m[6],p=m[8],y=m[9],x=m[10];return m!==f&&(f[0]=E,f[1]=D,f[2]=M,f[4]=$,f[5]=C,f[6]=g),f[8]=E*v+$*_+p,f[9]=D*v+C*_+y,f[10]=M*v+g*_+x,f},translation(m,S){const b=S??new t(12);return b[0]=1,b[1]=0,b[2]=0,b[4]=0,b[5]=1,b[6]=0,b[8]=m[0],b[9]=m[1],b[10]=1,b},transpose(m,S){const b=S??new t(12);if(b===m){let p;return p=m[1],m[1]=m[4],m[4]=p,p=m[2],m[2]=m[8],m[8]=p,p=m[6],m[6]=m[9],m[9]=p,b}const f=m[0],v=m[1],_=m[2],E=m[4],D=m[5],M=m[6],$=m[8],C=m[9],g=m[10];return b[0]=f,b[1]=E,b[2]=$,b[4]=v,b[5]=D,b[6]=C,b[8]=_,b[9]=M,b[10]=g,b},uniformScale(m,S,b){const f=b??new t(12);return f[0]=S*m[0],f[1]=S*m[1],f[2]=S*m[2],f[4]=S*m[4],f[5]=S*m[5],f[6]=S*m[6],m!==f&&(f[8]=m[8],f[9]=m[9],f[10]=m[10]),f},uniformScale3D(m,S,b){const f=b??new t(12);return f[0]=S*m[0],f[1]=S*m[1],f[2]=S*m[2],f[4]=S*m[4],f[5]=S*m[5],f[6]=S*m[6],f[8]=S*m[8],f[9]=S*m[9],f[10]=S*m[10],f},uniformScaling(m,S){const b=S??new t(12);return b[0]=m,b[1]=0,b[2]=0,b[4]=0,b[5]=m,b[6]=0,b[8]=0,b[9]=0,b[10]=1,b},uniformScaling3D(m,S){const b=S??new t(12);return b[0]=m,b[1]=0,b[2]=0,b[4]=0,b[5]=m,b[6]=0,b[8]=0,b[9]=0,b[10]=m,b}}})(s),Fd.set(s,e)),e}const Ud=new Map;function tC(s){let e=Ud.get(s);return e||(e=(t=>{const n=Hl(t);function r(f,v,_){const E=_??new t(16);return E[0]=f[0]*v,E[1]=f[1]*v,E[2]=f[2]*v,E[3]=f[3]*v,E[4]=f[4]*v,E[5]=f[5]*v,E[6]=f[6]*v,E[7]=f[7]*v,E[8]=f[8]*v,E[9]=f[9]*v,E[10]=f[10]*v,E[11]=f[11]*v,E[12]=f[12]*v,E[13]=f[13]*v,E[14]=f[14]*v,E[15]=f[15]*v,E}const i=r;function o(f,v){const _=v??new t(16);return _[0]=f[0],_[1]=f[1],_[2]=f[2],_[3]=f[3],_[4]=f[4],_[5]=f[5],_[6]=f[6],_[7]=f[7],_[8]=f[8],_[9]=f[9],_[10]=f[10],_[11]=f[11],_[12]=f[12],_[13]=f[13],_[14]=f[14],_[15]=f[15],_}const a=o;function l(f){const v=f??new t(16);return v[0]=1,v[1]=0,v[2]=0,v[3]=0,v[4]=0,v[5]=1,v[6]=0,v[7]=0,v[8]=0,v[9]=0,v[10]=1,v[11]=0,v[12]=0,v[13]=0,v[14]=0,v[15]=1,v}function u(f,v){const _=v??new t(16),E=f[0],D=f[1],M=f[2],$=f[3],C=f[4],g=f[5],p=f[6],y=f[7],x=f[8],I=f[9],N=f[10],L=f[11],W=f[12],X=f[13],V=f[14],Z=f[15],te=N*Z,oe=V*L,ce=p*Z,fe=V*y,We=p*L,je=N*y,Qe=M*Z,gt=V*$,yt=M*L,bt=N*$,Wt=M*y,Gt=p*$,Vt=x*X,qt=W*I,Ht=C*X,jt=W*g,Kt=C*I,Wo=x*g,Go=E*X,Vo=W*D,qo=E*I,Ho=x*D,jo=E*g,Ko=C*D,Nh=te*g+fe*I+We*X-(oe*g+ce*I+je*X),$h=oe*D+Qe*I+bt*X-(te*D+gt*I+yt*X),Dh=ce*D+gt*g+Wt*X-(fe*D+Qe*g+Gt*X),Oh=je*D+yt*g+Gt*I-(We*D+bt*g+Wt*I),wt=1/(E*Nh+C*$h+x*Dh+W*Oh);return _[0]=wt*Nh,_[1]=wt*$h,_[2]=wt*Dh,_[3]=wt*Oh,_[4]=wt*(oe*C+ce*x+je*W-(te*C+fe*x+We*W)),_[5]=wt*(te*E+gt*x+yt*W-(oe*E+Qe*x+bt*W)),_[6]=wt*(fe*E+Qe*C+Gt*W-(ce*E+gt*C+Wt*W)),_[7]=wt*(We*E+bt*C+Wt*x-(je*E+yt*C+Gt*x)),_[8]=wt*(Vt*y+jt*L+Kt*Z-(qt*y+Ht*L+Wo*Z)),_[9]=wt*(qt*$+Go*L+Ho*Z-(Vt*$+Vo*L+qo*Z)),_[10]=wt*(Ht*$+Vo*y+jo*Z-(jt*$+Go*y+Ko*Z)),_[11]=wt*(Wo*$+qo*y+Ko*L-(Kt*$+Ho*y+jo*L)),_[12]=wt*(Ht*N+Wo*V+qt*p-(Kt*V+Vt*p+jt*N)),_[13]=wt*(qo*V+Vt*M+Vo*N-(Go*N+Ho*V+qt*M)),_[14]=wt*(Go*p+Ko*V+jt*M-(jo*V+Ht*M+Vo*p)),_[15]=wt*(jo*N+Kt*M+Ho*p-(qo*p+Ko*N+Wo*M)),_}const c=u;function h(f,v,_){const E=_??new t(16),D=f[0],M=f[1],$=f[2],C=f[3],g=f[4],p=f[5],y=f[6],x=f[7],I=f[8],N=f[9],L=f[10],W=f[11],X=f[12],V=f[13],Z=f[14],te=f[15],oe=v[0],ce=v[1],fe=v[2],We=v[3],je=v[4],Qe=v[5],gt=v[6],yt=v[7],bt=v[8],Wt=v[9],Gt=v[10],Vt=v[11],qt=v[12],Ht=v[13],jt=v[14],Kt=v[15];return E[0]=D*oe+g*ce+I*fe+X*We,E[1]=M*oe+p*ce+N*fe+V*We,E[2]=$*oe+y*ce+L*fe+Z*We,E[3]=C*oe+x*ce+W*fe+te*We,E[4]=D*je+g*Qe+I*gt+X*yt,E[5]=M*je+p*Qe+N*gt+V*yt,E[6]=$*je+y*Qe+L*gt+Z*yt,E[7]=C*je+x*Qe+W*gt+te*yt,E[8]=D*bt+g*Wt+I*Gt+X*Vt,E[9]=M*bt+p*Wt+N*Gt+V*Vt,E[10]=$*bt+y*Wt+L*Gt+Z*Vt,E[11]=C*bt+x*Wt+W*Gt+te*Vt,E[12]=D*qt+g*Ht+I*jt+X*Kt,E[13]=M*qt+p*Ht+N*jt+V*Kt,E[14]=$*qt+y*Ht+L*jt+Z*Kt,E[15]=C*qt+x*Ht+W*jt+te*Kt,E}const d=h,w=n.create(),k=n.create(),A=n.create();function m(f,v,_){const E=_??new t(16);let D=f[0],M=f[1],$=f[2];const C=Math.sqrt(D*D+M*M+$*$);D/=C,M/=C,$/=C;const g=D*D,p=M*M,y=$*$,x=Math.cos(v),I=Math.sin(v),N=1-x;return E[0]=g+(1-g)*x,E[1]=D*M*N+$*I,E[2]=D*$*N-M*I,E[3]=0,E[4]=D*M*N-$*I,E[5]=p+(1-p)*x,E[6]=M*$*N+D*I,E[7]=0,E[8]=D*$*N+M*I,E[9]=M*$*N-D*I,E[10]=y+(1-y)*x,E[11]=0,E[12]=0,E[13]=0,E[14]=0,E[15]=1,E}const S=m;function b(f,v,_,E){const D=E??new t(16);let M=v[0],$=v[1],C=v[2];const g=Math.sqrt(M*M+$*$+C*C);M/=g,$/=g,C/=g;const p=M*M,y=$*$,x=C*C,I=Math.cos(_),N=Math.sin(_),L=1-I,W=p+(1-p)*I,X=M*$*L+C*N,V=M*C*L-$*N,Z=M*$*L-C*N,te=y+(1-y)*I,oe=$*C*L+M*N,ce=M*C*L+$*N,fe=$*C*L-M*N,We=x+(1-x)*I,je=f[0],Qe=f[1],gt=f[2],yt=f[3],bt=f[4],Wt=f[5],Gt=f[6],Vt=f[7],qt=f[8],Ht=f[9],jt=f[10],Kt=f[11];return D[0]=W*je+X*bt+V*qt,D[1]=W*Qe+X*Wt+V*Ht,D[2]=W*gt+X*Gt+V*jt,D[3]=W*yt+X*Vt+V*Kt,D[4]=Z*je+te*bt+oe*qt,D[5]=Z*Qe+te*Wt+oe*Ht,D[6]=Z*gt+te*Gt+oe*jt,D[7]=Z*yt+te*Vt+oe*Kt,D[8]=ce*je+fe*bt+We*qt,D[9]=ce*Qe+fe*Wt+We*Ht,D[10]=ce*gt+fe*Gt+We*jt,D[11]=ce*yt+fe*Vt+We*Kt,f!==D&&(D[12]=f[12],D[13]=f[13],D[14]=f[14],D[15]=f[15]),D}return{add(f,v,_){const E=_??new t(16);return E[0]=f[0]+v[0],E[1]=f[1]+v[1],E[2]=f[2]+v[2],E[3]=f[3]+v[3],E[4]=f[4]+v[4],E[5]=f[5]+v[5],E[6]=f[6]+v[6],E[7]=f[7]+v[7],E[8]=f[8]+v[8],E[9]=f[9]+v[9],E[10]=f[10]+v[10],E[11]=f[11]+v[11],E[12]=f[12]+v[12],E[13]=f[13]+v[13],E[14]=f[14]+v[14],E[15]=f[15]+v[15],E},aim(f,v,_,E){const D=E??new t(16);return n.normalize(n.subtract(v,f,A),A),n.normalize(n.cross(_,A,w),w),n.normalize(n.cross(A,w,k),k),D[0]=w[0],D[1]=w[1],D[2]=w[2],D[3]=0,D[4]=k[0],D[5]=k[1],D[6]=k[2],D[7]=0,D[8]=A[0],D[9]=A[1],D[10]=A[2],D[11]=0,D[12]=f[0],D[13]=f[1],D[14]=f[2],D[15]=1,D},axisRotate:b,axisRotation:m,cameraAim(f,v,_,E){const D=E??new t(16);return n.normalize(n.subtract(f,v,A),A),n.normalize(n.cross(_,A,w),w),n.normalize(n.cross(A,w,k),k),D[0]=w[0],D[1]=w[1],D[2]=w[2],D[3]=0,D[4]=k[0],D[5]=k[1],D[6]=k[2],D[7]=0,D[8]=A[0],D[9]=A[1],D[10]=A[2],D[11]=0,D[12]=f[0],D[13]=f[1],D[14]=f[2],D[15]=1,D},clone:a,copy:o,create(f,v,_,E,D,M,$,C,g,p,y,x,I,N,L,W){const X=new t(16);return f!==void 0&&(X[0]=f,v!==void 0&&(X[1]=v,_!==void 0&&(X[2]=_,E!==void 0&&(X[3]=E,D!==void 0&&(X[4]=D,M!==void 0&&(X[5]=M,$!==void 0&&(X[6]=$,C!==void 0&&(X[7]=C,g!==void 0&&(X[8]=g,p!==void 0&&(X[9]=p,y!==void 0&&(X[10]=y,x!==void 0&&(X[11]=x,I!==void 0&&(X[12]=I,N!==void 0&&(X[13]=N,L!==void 0&&(X[14]=L,W!==void 0&&(X[15]=W)))))))))))))))),X},determinant(f){const v=f[0],_=f[1],E=f[2],D=f[3],M=f[4],$=f[5],C=f[6],g=f[7],p=f[8],y=f[9],x=f[10],I=f[11],N=f[12],L=f[13],W=f[14],X=f[15],V=x*X,Z=W*I,te=C*X,oe=W*g,ce=C*I,fe=x*g,We=E*X,je=W*D,Qe=E*I,gt=x*D,yt=E*g,bt=C*D;return v*(V*$+oe*y+ce*L-(Z*$+te*y+fe*L))+M*(Z*_+We*y+gt*L-(V*_+je*y+Qe*L))+p*(te*_+je*$+yt*L-(oe*_+We*$+bt*L))+N*(fe*_+Qe*$+bt*y-(ce*_+gt*$+yt*y))},equals(f,v){return f[0]===v[0]&&f[1]===v[1]&&f[2]===v[2]&&f[3]===v[3]&&f[4]===v[4]&&f[5]===v[5]&&f[6]===v[6]&&f[7]===v[7]&&f[8]===v[8]&&f[9]===v[9]&&f[10]===v[10]&&f[11]===v[11]&&f[12]===v[12]&&f[13]===v[13]&&f[14]===v[14]&&f[15]===v[15]},equalsApproximately(f,v){return Math.abs(f[0]-v[0])<be&&Math.abs(f[1]-v[1])<be&&Math.abs(f[2]-v[2])<be&&Math.abs(f[3]-v[3])<be&&Math.abs(f[4]-v[4])<be&&Math.abs(f[5]-v[5])<be&&Math.abs(f[6]-v[6])<be&&Math.abs(f[7]-v[7])<be&&Math.abs(f[8]-v[8])<be&&Math.abs(f[9]-v[9])<be&&Math.abs(f[10]-v[10])<be&&Math.abs(f[11]-v[11])<be&&Math.abs(f[12]-v[12])<be&&Math.abs(f[13]-v[13])<be&&Math.abs(f[14]-v[14])<be&&Math.abs(f[15]-v[15])<be},fromMat3(f,v){const _=v??new t(16);return _[0]=f[0],_[1]=f[1],_[2]=f[2],_[3]=0,_[4]=f[4],_[5]=f[5],_[6]=f[6],_[7]=0,_[8]=f[8],_[9]=f[9],_[10]=f[10],_[11]=0,_[12]=0,_[13]=0,_[14]=0,_[15]=1,_},fromQuat(f,v){const _=v??new t(16),E=f[0],D=f[1],M=f[2],$=f[3],C=E+E,g=D+D,p=M+M,y=E*C,x=D*C,I=D*g,N=M*C,L=M*g,W=M*p,X=$*C,V=$*g,Z=$*p;return _[0]=1-I-W,_[1]=x+Z,_[2]=N-V,_[3]=0,_[4]=x-Z,_[5]=1-y-W,_[6]=L+X,_[7]=0,_[8]=N+V,_[9]=L-X,_[10]=1-y-I,_[11]=0,_[12]=0,_[13]=0,_[14]=0,_[15]=1,_},frustum(f,v,_,E,D,M,$){const C=$??new t(16),g=v-f,p=E-_,y=D-M;return C[0]=2*D/g,C[1]=0,C[2]=0,C[3]=0,C[4]=0,C[5]=2*D/p,C[6]=0,C[7]=0,C[8]=(f+v)/g,C[9]=(E+_)/p,C[10]=M/y,C[11]=-1,C[12]=0,C[13]=0,C[14]=D*M/y,C[15]=0,C},frustumReverseZ(f,v,_,E,D,M=1/0,$){const C=$??new t(16),g=v-f,p=E-_;if(C[0]=2*D/g,C[1]=0,C[2]=0,C[3]=0,C[4]=0,C[5]=2*D/p,C[6]=0,C[7]=0,C[8]=(f+v)/g,C[9]=(E+_)/p,C[11]=-1,C[12]=0,C[13]=0,C[15]=0,M===1/0)C[10]=0,C[14]=D;else{const y=1/(M-D);C[10]=D*y,C[14]=M*D*y}return C},getAxis(f,v,_){const E=_??n.create(),D=4*v;return E[0]=f[D+0],E[1]=f[D+1],E[2]=f[D+2],E},getScaling(f,v){const _=v??n.create(),E=f[0],D=f[1],M=f[2],$=f[4],C=f[5],g=f[6],p=f[8],y=f[9],x=f[10];return _[0]=Math.sqrt(E*E+D*D+M*M),_[1]=Math.sqrt($*$+C*C+g*g),_[2]=Math.sqrt(p*p+y*y+x*x),_},getTranslation(f,v){const _=v??n.create();return _[0]=f[12],_[1]=f[13],_[2]=f[14],_},identity:l,inverse:u,invert:c,lookAt(f,v,_,E){const D=E??new t(16);return n.normalize(n.subtract(f,v,A),A),n.normalize(n.cross(_,A,w),w),n.normalize(n.cross(A,w,k),k),D[0]=w[0],D[1]=k[0],D[2]=A[0],D[3]=0,D[4]=w[1],D[5]=k[1],D[6]=A[1],D[7]=0,D[8]=w[2],D[9]=k[2],D[10]=A[2],D[11]=0,D[12]=-(w[0]*f[0]+w[1]*f[1]+w[2]*f[2]),D[13]=-(k[0]*f[0]+k[1]*f[1]+k[2]*f[2]),D[14]=-(A[0]*f[0]+A[1]*f[1]+A[2]*f[2]),D[15]=1,D},mul:d,mulScalar:i,multiply:h,multiplyScalar:r,negate(f,v){const _=v??new t(16);return _[0]=-f[0],_[1]=-f[1],_[2]=-f[2],_[3]=-f[3],_[4]=-f[4],_[5]=-f[5],_[6]=-f[6],_[7]=-f[7],_[8]=-f[8],_[9]=-f[9],_[10]=-f[10],_[11]=-f[11],_[12]=-f[12],_[13]=-f[13],_[14]=-f[14],_[15]=-f[15],_},ortho(f,v,_,E,D,M,$){const C=$??new t(16);return C[0]=2/(v-f),C[1]=0,C[2]=0,C[3]=0,C[4]=0,C[5]=2/(E-_),C[6]=0,C[7]=0,C[8]=0,C[9]=0,C[10]=1/(D-M),C[11]=0,C[12]=(v+f)/(f-v),C[13]=(E+_)/(_-E),C[14]=D/(D-M),C[15]=1,C},perspective(f,v,_,E,D){const M=D??new t(16),$=Math.tan(.5*Math.PI-.5*f);if(M[0]=$/v,M[1]=0,M[2]=0,M[3]=0,M[4]=0,M[5]=$,M[6]=0,M[7]=0,M[8]=0,M[9]=0,M[11]=-1,M[12]=0,M[13]=0,M[15]=0,Number.isFinite(E)){const C=1/(_-E);M[10]=E*C,M[14]=E*_*C}else M[10]=-1,M[14]=-_;return M},perspectiveReverseZ(f,v,_,E=1/0,D){const M=D??new t(16),$=1/Math.tan(.5*f);if(M[0]=$/v,M[1]=0,M[2]=0,M[3]=0,M[4]=0,M[5]=$,M[6]=0,M[7]=0,M[8]=0,M[9]=0,M[11]=-1,M[12]=0,M[13]=0,M[15]=0,E===1/0)M[10]=0,M[14]=_;else{const C=1/(E-_);M[10]=_*C,M[14]=E*_*C}return M},rotate:b,rotateX(f,v,_){const E=_??new t(16),D=f[4],M=f[5],$=f[6],C=f[7],g=f[8],p=f[9],y=f[10],x=f[11],I=Math.cos(v),N=Math.sin(v);return E[4]=I*D+N*g,E[5]=I*M+N*p,E[6]=I*$+N*y,E[7]=I*C+N*x,E[8]=I*g-N*D,E[9]=I*p-N*M,E[10]=I*y-N*$,E[11]=I*x-N*C,f!==E&&(E[0]=f[0],E[1]=f[1],E[2]=f[2],E[3]=f[3],E[12]=f[12],E[13]=f[13],E[14]=f[14],E[15]=f[15]),E},rotateY(f,v,_){const E=_??new t(16),D=f[0],M=f[1],$=f[2],C=f[3],g=f[8],p=f[9],y=f[10],x=f[11],I=Math.cos(v),N=Math.sin(v);return E[0]=I*D-N*g,E[1]=I*M-N*p,E[2]=I*$-N*y,E[3]=I*C-N*x,E[8]=I*g+N*D,E[9]=I*p+N*M,E[10]=I*y+N*$,E[11]=I*x+N*C,f!==E&&(E[4]=f[4],E[5]=f[5],E[6]=f[6],E[7]=f[7],E[12]=f[12],E[13]=f[13],E[14]=f[14],E[15]=f[15]),E},rotateZ(f,v,_){const E=_??new t(16),D=f[0],M=f[1],$=f[2],C=f[3],g=f[4],p=f[5],y=f[6],x=f[7],I=Math.cos(v),N=Math.sin(v);return E[0]=I*D+N*g,E[1]=I*M+N*p,E[2]=I*$+N*y,E[3]=I*C+N*x,E[4]=I*g-N*D,E[5]=I*p-N*M,E[6]=I*y-N*$,E[7]=I*x-N*C,f!==E&&(E[8]=f[8],E[9]=f[9],E[10]=f[10],E[11]=f[11],E[12]=f[12],E[13]=f[13],E[14]=f[14],E[15]=f[15]),E},rotation:S,rotationX(f,v){const _=v??new t(16),E=Math.cos(f),D=Math.sin(f);return _[0]=1,_[1]=0,_[2]=0,_[3]=0,_[4]=0,_[5]=E,_[6]=D,_[7]=0,_[8]=0,_[9]=-D,_[10]=E,_[11]=0,_[12]=0,_[13]=0,_[14]=0,_[15]=1,_},rotationY(f,v){const _=v??new t(16),E=Math.cos(f),D=Math.sin(f);return _[0]=E,_[1]=0,_[2]=-D,_[3]=0,_[4]=0,_[5]=1,_[6]=0,_[7]=0,_[8]=D,_[9]=0,_[10]=E,_[11]=0,_[12]=0,_[13]=0,_[14]=0,_[15]=1,_},rotationZ(f,v){const _=v??new t(16),E=Math.cos(f),D=Math.sin(f);return _[0]=E,_[1]=D,_[2]=0,_[3]=0,_[4]=-D,_[5]=E,_[6]=0,_[7]=0,_[8]=0,_[9]=0,_[10]=1,_[11]=0,_[12]=0,_[13]=0,_[14]=0,_[15]=1,_},scale(f,v,_){const E=_??new t(16),D=v[0],M=v[1],$=v[2];return E[0]=D*f[0],E[1]=D*f[1],E[2]=D*f[2],E[3]=D*f[3],E[4]=M*f[4],E[5]=M*f[5],E[6]=M*f[6],E[7]=M*f[7],E[8]=$*f[8],E[9]=$*f[9],E[10]=$*f[10],E[11]=$*f[11],f!==E&&(E[12]=f[12],E[13]=f[13],E[14]=f[14],E[15]=f[15]),E},scaling(f,v){const _=v??new t(16);return _[0]=f[0],_[1]=0,_[2]=0,_[3]=0,_[4]=0,_[5]=f[1],_[6]=0,_[7]=0,_[8]=0,_[9]=0,_[10]=f[2],_[11]=0,_[12]=0,_[13]=0,_[14]=0,_[15]=1,_},set(f,v,_,E,D,M,$,C,g,p,y,x,I,N,L,W,X){const V=X??new t(16);return V[0]=f,V[1]=v,V[2]=_,V[3]=E,V[4]=D,V[5]=M,V[6]=$,V[7]=C,V[8]=g,V[9]=p,V[10]=y,V[11]=x,V[12]=I,V[13]=N,V[14]=L,V[15]=W,V},setAxis(f,v,_,E){const D=E===f?E:o(f,E),M=4*_;return D[M+0]=v[0],D[M+1]=v[1],D[M+2]=v[2],D},setTranslation(f,v,_){const E=_??l();return f!==E&&(E[0]=f[0],E[1]=f[1],E[2]=f[2],E[3]=f[3],E[4]=f[4],E[5]=f[5],E[6]=f[6],E[7]=f[7],E[8]=f[8],E[9]=f[9],E[10]=f[10],E[11]=f[11]),E[12]=v[0],E[13]=v[1],E[14]=v[2],E[15]=1,E},translate(f,v,_){const E=_??new t(16),D=v[0],M=v[1],$=v[2],C=f[0],g=f[1],p=f[2],y=f[3],x=f[4],I=f[5],N=f[6],L=f[7],W=f[8],X=f[9],V=f[10],Z=f[11],te=f[12],oe=f[13],ce=f[14],fe=f[15];return f!==E&&(E[0]=C,E[1]=g,E[2]=p,E[3]=y,E[4]=x,E[5]=I,E[6]=N,E[7]=L,E[8]=W,E[9]=X,E[10]=V,E[11]=Z),E[12]=C*D+x*M+W*$+te,E[13]=g*D+I*M+X*$+oe,E[14]=p*D+N*M+V*$+ce,E[15]=y*D+L*M+Z*$+fe,E},translation(f,v){const _=v??new t(16);return _[0]=1,_[1]=0,_[2]=0,_[3]=0,_[4]=0,_[5]=1,_[6]=0,_[7]=0,_[8]=0,_[9]=0,_[10]=1,_[11]=0,_[12]=f[0],_[13]=f[1],_[14]=f[2],_[15]=1,_},transpose(f,v){const _=v??new t(16);if(_===f){let te;return te=f[1],f[1]=f[4],f[4]=te,te=f[2],f[2]=f[8],f[8]=te,te=f[3],f[3]=f[12],f[12]=te,te=f[6],f[6]=f[9],f[9]=te,te=f[7],f[7]=f[13],f[13]=te,te=f[11],f[11]=f[14],f[14]=te,_}const E=f[0],D=f[1],M=f[2],$=f[3],C=f[4],g=f[5],p=f[6],y=f[7],x=f[8],I=f[9],N=f[10],L=f[11],W=f[12],X=f[13],V=f[14],Z=f[15];return _[0]=E,_[1]=C,_[2]=x,_[3]=W,_[4]=D,_[5]=g,_[6]=I,_[7]=X,_[8]=M,_[9]=p,_[10]=N,_[11]=V,_[12]=$,_[13]=y,_[14]=L,_[15]=Z,_},uniformScale(f,v,_){const E=_??new t(16);return E[0]=v*f[0],E[1]=v*f[1],E[2]=v*f[2],E[3]=v*f[3],E[4]=v*f[4],E[5]=v*f[5],E[6]=v*f[6],E[7]=v*f[7],E[8]=v*f[8],E[9]=v*f[9],E[10]=v*f[10],E[11]=v*f[11],f!==E&&(E[12]=f[12],E[13]=f[13],E[14]=f[14],E[15]=f[15]),E},uniformScaling(f,v){const _=v??new t(16);return _[0]=f,_[1]=0,_[2]=0,_[3]=0,_[4]=0,_[5]=f,_[6]=0,_[7]=0,_[8]=0,_[9]=0,_[10]=f,_[11]=0,_[12]=0,_[13]=0,_[14]=0,_[15]=1,_}}})(s),Ud.set(s,e)),e}const zd=new Map;function nC(s){let e=zd.get(s);return e||(e=(t=>{const n=Hl(t);function r(g,p,y,x){const I=new t(4);return g!==void 0&&(I[0]=g,p!==void 0&&(I[1]=p,y!==void 0&&(I[2]=y,x!==void 0&&(I[3]=x)))),I}const i=r;function o(g,p,y){const x=y??new t(4),I=.5*p,N=Math.sin(I);return x[0]=N*g[0],x[1]=N*g[1],x[2]=N*g[2],x[3]=Math.cos(I),x}function a(g,p,y){const x=y??new t(4),I=g[0],N=g[1],L=g[2],W=g[3],X=p[0],V=p[1],Z=p[2],te=p[3];return x[0]=I*te+W*X+N*Z-L*V,x[1]=N*te+W*V+L*X-I*Z,x[2]=L*te+W*Z+I*V-N*X,x[3]=W*te-I*X-N*V-L*Z,x}const l=a;function u(g,p,y,x){const I=x??new t(4),N=g[0],L=g[1],W=g[2],X=g[3];let V,Z,te=p[0],oe=p[1],ce=p[2],fe=p[3],We=N*te+L*oe+W*ce+X*fe;if(We<0&&(We=-We,te=-te,oe=-oe,ce=-ce,fe=-fe),1-We>be){const je=Math.acos(We),Qe=Math.sin(je);V=Math.sin((1-y)*je)/Qe,Z=Math.sin(y*je)/Qe}else V=1-y,Z=y;return I[0]=V*N+Z*te,I[1]=V*L+Z*oe,I[2]=V*W+Z*ce,I[3]=V*X+Z*fe,I}function c(g,p){const y=p??new t(4);return y[0]=g[0],y[1]=g[1],y[2]=g[2],y[3]=g[3],y}const h=c;function d(g,p,y){const x=y??new t(4);return x[0]=g[0]-p[0],x[1]=g[1]-p[1],x[2]=g[2]-p[2],x[3]=g[3]-p[3],x}const w=d;function k(g,p,y){const x=y??new t(4);return x[0]=g[0]*p,x[1]=g[1]*p,x[2]=g[2]*p,x[3]=g[3]*p,x}const A=k;function m(g,p){return g[0]*p[0]+g[1]*p[1]+g[2]*p[2]+g[3]*p[3]}function S(g){const p=g[0],y=g[1],x=g[2],I=g[3];return Math.sqrt(p*p+y*y+x*x+I*I)}const b=S;function f(g){const p=g[0],y=g[1],x=g[2],I=g[3];return p*p+y*y+x*x+I*I}const v=f;function _(g,p){const y=p??new t(4),x=g[0],I=g[1],N=g[2],L=g[3],W=Math.sqrt(x*x+I*I+N*N+L*L);return W>1e-5?(y[0]=x/W,y[1]=I/W,y[2]=N/W,y[3]=L/W):(y[0]=0,y[1]=0,y[2]=0,y[3]=1),y}const E=n.create(),D=n.create(),M=n.create(),$=new t(4),C=new t(4);return{create:r,fromValues:i,set(g,p,y,x,I){const N=I??new t(4);return N[0]=g,N[1]=p,N[2]=y,N[3]=x,N},fromAxisAngle:o,toAxisAngle(g,p){const y=p??n.create(3),x=2*Math.acos(g[3]),I=Math.sin(.5*x);return I>be?(y[0]=g[0]/I,y[1]=g[1]/I,y[2]=g[2]/I):(y[0]=1,y[1]=0,y[2]=0),{angle:x,axis:y}},angle(g,p){const y=m(g,p);return Math.acos(2*y*y-1)},multiply:a,mul:l,rotateX(g,p,y){const x=y??new t(4),I=.5*p,N=g[0],L=g[1],W=g[2],X=g[3],V=Math.sin(I),Z=Math.cos(I);return x[0]=N*Z+X*V,x[1]=L*Z+W*V,x[2]=W*Z-L*V,x[3]=X*Z-N*V,x},rotateY(g,p,y){const x=y??new t(4),I=.5*p,N=g[0],L=g[1],W=g[2],X=g[3],V=Math.sin(I),Z=Math.cos(I);return x[0]=N*Z-W*V,x[1]=L*Z+X*V,x[2]=W*Z+N*V,x[3]=X*Z-L*V,x},rotateZ(g,p,y){const x=y??new t(4),I=.5*p,N=g[0],L=g[1],W=g[2],X=g[3],V=Math.sin(I),Z=Math.cos(I);return x[0]=N*Z+L*V,x[1]=L*Z-N*V,x[2]=W*Z+X*V,x[3]=X*Z-W*V,x},slerp:u,inverse(g,p){const y=p??new t(4),x=g[0],I=g[1],N=g[2],L=g[3],W=x*x+I*I+N*N+L*L,X=W?1/W:0;return y[0]=-x*X,y[1]=-I*X,y[2]=-N*X,y[3]=L*X,y},conjugate(g,p){const y=p??new t(4);return y[0]=-g[0],y[1]=-g[1],y[2]=-g[2],y[3]=g[3],y},fromMat(g,p){const y=p??new t(4),x=g[0]+g[5]+g[10];if(x>0){const I=Math.sqrt(x+1);y[3]=.5*I;const N=.5/I;y[0]=(g[6]-g[9])*N,y[1]=(g[8]-g[2])*N,y[2]=(g[1]-g[4])*N}else{let I=0;g[5]>g[0]&&(I=1),g[10]>g[4*I+I]&&(I=2);const N=(I+1)%3,L=(I+2)%3,W=Math.sqrt(g[4*I+I]-g[4*N+N]-g[4*L+L]+1);y[I]=.5*W;const X=.5/W;y[3]=(g[4*N+L]-g[4*L+N])*X,y[N]=(g[4*N+I]+g[4*I+N])*X,y[L]=(g[4*L+I]+g[4*I+L])*X}return y},fromEuler(g,p,y,x,I){const N=I??new t(4),L=.5*g,W=.5*p,X=.5*y,V=Math.sin(L),Z=Math.cos(L),te=Math.sin(W),oe=Math.cos(W),ce=Math.sin(X),fe=Math.cos(X);switch(x){case"xyz":N[0]=V*oe*fe+Z*te*ce,N[1]=Z*te*fe-V*oe*ce,N[2]=Z*oe*ce+V*te*fe,N[3]=Z*oe*fe-V*te*ce;break;case"xzy":N[0]=V*oe*fe-Z*te*ce,N[1]=Z*te*fe-V*oe*ce,N[2]=Z*oe*ce+V*te*fe,N[3]=Z*oe*fe+V*te*ce;break;case"yxz":N[0]=V*oe*fe+Z*te*ce,N[1]=Z*te*fe-V*oe*ce,N[2]=Z*oe*ce-V*te*fe,N[3]=Z*oe*fe+V*te*ce;break;case"yzx":N[0]=V*oe*fe+Z*te*ce,N[1]=Z*te*fe+V*oe*ce,N[2]=Z*oe*ce-V*te*fe,N[3]=Z*oe*fe-V*te*ce;break;case"zxy":N[0]=V*oe*fe-Z*te*ce,N[1]=Z*te*fe+V*oe*ce,N[2]=Z*oe*ce+V*te*fe,N[3]=Z*oe*fe-V*te*ce;break;case"zyx":N[0]=V*oe*fe-Z*te*ce,N[1]=Z*te*fe+V*oe*ce,N[2]=Z*oe*ce-V*te*fe,N[3]=Z*oe*fe+V*te*ce;break;default:throw Error("Unknown rotation order: "+x)}return N},copy:c,clone:h,add(g,p,y){const x=y??new t(4);return x[0]=g[0]+p[0],x[1]=g[1]+p[1],x[2]=g[2]+p[2],x[3]=g[3]+p[3],x},subtract:d,sub:w,mulScalar:k,scale:A,divScalar(g,p,y){const x=y??new t(4);return x[0]=g[0]/p,x[1]=g[1]/p,x[2]=g[2]/p,x[3]=g[3]/p,x},dot:m,lerp(g,p,y,x){const I=x??new t(4);return I[0]=g[0]+y*(p[0]-g[0]),I[1]=g[1]+y*(p[1]-g[1]),I[2]=g[2]+y*(p[2]-g[2]),I[3]=g[3]+y*(p[3]-g[3]),I},length:S,len:b,lengthSq:f,lenSq:v,normalize:_,equalsApproximately(g,p){return Math.abs(g[0]-p[0])<be&&Math.abs(g[1]-p[1])<be&&Math.abs(g[2]-p[2])<be&&Math.abs(g[3]-p[3])<be},equals(g,p){return g[0]===p[0]&&g[1]===p[1]&&g[2]===p[2]&&g[3]===p[3]},identity(g){const p=g??new t(4);return p[0]=0,p[1]=0,p[2]=0,p[3]=1,p},rotationTo(g,p,y){const x=y??new t(4),I=n.dot(g,p);return I<-.999999?(n.cross(D,g,E),n.len(E)<1e-6&&n.cross(M,g,E),n.normalize(E,E),o(E,Math.PI,x),x):I>.999999?(x[0]=0,x[1]=0,x[2]=0,x[3]=1,x):(n.cross(g,p,E),x[0]=E[0],x[1]=E[1],x[2]=E[2],x[3]=1+I,_(x,x))},sqlerp(g,p,y,x,I,N){const L=N??new t(4);return u(g,x,I,$),u(p,y,I,C),u($,C,2*I*(1-I),L),L}}})(s),zd.set(s,e)),e}const Wd=new Map;function sC(s){let e=Wd.get(s);return e||(e=(t=>{function n(g,p,y,x){const I=new t(4);return g!==void 0&&(I[0]=g,p!==void 0&&(I[1]=p,y!==void 0&&(I[2]=y,x!==void 0&&(I[3]=x)))),I}function r(g,p,y){const x=y??new t(4);return x[0]=g[0]-p[0],x[1]=g[1]-p[1],x[2]=g[2]-p[2],x[3]=g[3]-p[3],x}const i=r;function o(g,p,y,x){const I=x??new t(4);return I[0]=g[0]+y*(p[0]-g[0]),I[1]=g[1]+y*(p[1]-g[1]),I[2]=g[2]+y*(p[2]-g[2]),I[3]=g[3]+y*(p[3]-g[3]),I}function a(g,p,y){const x=y??new t(4);return x[0]=g[0]*p,x[1]=g[1]*p,x[2]=g[2]*p,x[3]=g[3]*p,x}const l=a;function u(g,p){const y=p??new t(4);return y[0]=1/g[0],y[1]=1/g[1],y[2]=1/g[2],y[3]=1/g[3],y}const c=u;function h(g){const p=g[0],y=g[1],x=g[2],I=g[3];return Math.sqrt(p*p+y*y+x*x+I*I)}const d=h;function w(g){const p=g[0],y=g[1],x=g[2],I=g[3];return p*p+y*y+x*x+I*I}const k=w;function A(g,p){const y=g[0]-p[0],x=g[1]-p[1],I=g[2]-p[2],N=g[3]-p[3];return Math.sqrt(y*y+x*x+I*I+N*N)}const m=A;function S(g,p){const y=g[0]-p[0],x=g[1]-p[1],I=g[2]-p[2],N=g[3]-p[3];return y*y+x*x+I*I+N*N}const b=S;function f(g,p){const y=p??new t(4),x=g[0],I=g[1],N=g[2],L=g[3],W=Math.sqrt(x*x+I*I+N*N+L*L);return W>1e-5?(y[0]=x/W,y[1]=I/W,y[2]=N/W,y[3]=L/W):(y[0]=0,y[1]=0,y[2]=0,y[3]=0),y}function v(g,p){const y=p??new t(4);return y[0]=g[0],y[1]=g[1],y[2]=g[2],y[3]=g[3],y}const _=v;function E(g,p,y){const x=y??new t(4);return x[0]=g[0]*p[0],x[1]=g[1]*p[1],x[2]=g[2]*p[2],x[3]=g[3]*p[3],x}const D=E;function M(g,p,y){const x=y??new t(4);return x[0]=g[0]/p[0],x[1]=g[1]/p[1],x[2]=g[2]/p[2],x[3]=g[3]/p[3],x}const $=M;function C(g,p,y){const x=y??new t(4);return f(g,x),a(x,p,x)}return{create:n,fromValues:n,set(g,p,y,x,I){const N=I??new t(4);return N[0]=g,N[1]=p,N[2]=y,N[3]=x,N},ceil(g,p){const y=p??new t(4);return y[0]=Math.ceil(g[0]),y[1]=Math.ceil(g[1]),y[2]=Math.ceil(g[2]),y[3]=Math.ceil(g[3]),y},floor(g,p){const y=p??new t(4);return y[0]=Math.floor(g[0]),y[1]=Math.floor(g[1]),y[2]=Math.floor(g[2]),y[3]=Math.floor(g[3]),y},round(g,p){const y=p??new t(4);return y[0]=Math.round(g[0]),y[1]=Math.round(g[1]),y[2]=Math.round(g[2]),y[3]=Math.round(g[3]),y},clamp(g,p=0,y=1,x){const I=x??new t(4);return I[0]=Math.min(y,Math.max(p,g[0])),I[1]=Math.min(y,Math.max(p,g[1])),I[2]=Math.min(y,Math.max(p,g[2])),I[3]=Math.min(y,Math.max(p,g[3])),I},add(g,p,y){const x=y??new t(4);return x[0]=g[0]+p[0],x[1]=g[1]+p[1],x[2]=g[2]+p[2],x[3]=g[3]+p[3],x},addScaled(g,p,y,x){const I=x??new t(4);return I[0]=g[0]+p[0]*y,I[1]=g[1]+p[1]*y,I[2]=g[2]+p[2]*y,I[3]=g[3]+p[3]*y,I},subtract:r,sub:i,equalsApproximately(g,p){return Math.abs(g[0]-p[0])<be&&Math.abs(g[1]-p[1])<be&&Math.abs(g[2]-p[2])<be&&Math.abs(g[3]-p[3])<be},equals(g,p){return g[0]===p[0]&&g[1]===p[1]&&g[2]===p[2]&&g[3]===p[3]},lerp:o,lerpV(g,p,y,x){const I=x??new t(4);return I[0]=g[0]+y[0]*(p[0]-g[0]),I[1]=g[1]+y[1]*(p[1]-g[1]),I[2]=g[2]+y[2]*(p[2]-g[2]),I[3]=g[3]+y[3]*(p[3]-g[3]),I},max(g,p,y){const x=y??new t(4);return x[0]=Math.max(g[0],p[0]),x[1]=Math.max(g[1],p[1]),x[2]=Math.max(g[2],p[2]),x[3]=Math.max(g[3],p[3]),x},min(g,p,y){const x=y??new t(4);return x[0]=Math.min(g[0],p[0]),x[1]=Math.min(g[1],p[1]),x[2]=Math.min(g[2],p[2]),x[3]=Math.min(g[3],p[3]),x},mulScalar:a,scale:l,divScalar(g,p,y){const x=y??new t(4);return x[0]=g[0]/p,x[1]=g[1]/p,x[2]=g[2]/p,x[3]=g[3]/p,x},inverse:u,invert:c,dot(g,p){return g[0]*p[0]+g[1]*p[1]+g[2]*p[2]+g[3]*p[3]},length:h,len:d,lengthSq:w,lenSq:k,distance:A,dist:m,distanceSq:S,distSq:b,normalize:f,negate(g,p){const y=p??new t(4);return y[0]=-g[0],y[1]=-g[1],y[2]=-g[2],y[3]=-g[3],y},copy:v,clone:_,multiply:E,mul:D,divide:M,div:$,zero(g){const p=g??new t(4);return p[0]=0,p[1]=0,p[2]=0,p[3]=0,p},transformMat4(g,p,y){const x=y??new t(4),I=g[0],N=g[1],L=g[2],W=g[3];return x[0]=p[0]*I+p[4]*N+p[8]*L+p[12]*W,x[1]=p[1]*I+p[5]*N+p[9]*L+p[13]*W,x[2]=p[2]*I+p[6]*N+p[10]*L+p[14]*W,x[3]=p[3]*I+p[7]*N+p[11]*L+p[15]*W,x},setLength:C,truncate(g,p,y){const x=y??new t(4);return h(g)>p?C(g,p,x):v(g,x)},midpoint(g,p,y){return o(g,p,.5,y??new t(4))}}})(s),Wd.set(s,e)),e}function lc(s,e,t,n,r,i){return{mat3:eC(s),mat4:tC(e),quat:nC(t),vec2:Vy(n),vec3:Hl(r),vec4:sC(i)}}const{mat3:vl,mat4:Gd,vec2:xu}=lc(Float32Array,Float32Array,Float32Array,Float32Array,Float32Array,Float32Array);lc(Float64Array,Float64Array,Float64Array,Float64Array,Float64Array,Float64Array),lc(JA,Array,Array,Array,Array,Array);var $a,as,ji,uo,Hs,Ge,js,Ki,Qr,Jr,vn,_n,at,wn,Vr,Ss,Un,Ks,Xi,Yi,Zi,qr,zn,ls,uc,cc,hc,qy;class rC extends zy{constructor(e,t,n,r){super(e,t,"Render"),ee(this,ls),ee(this,$a),ee(this,as,!1),ee(this,ji,vl.identity()),ee(this,uo,new Float32Array(3)),ee(this,Hs,!1),ee(this,Ge),ee(this,js),ee(this,Ki,Gd.identity()),ee(this,Qr),ee(this,Jr),ee(this,vn),ee(this,_n),ee(this,at),ee(this,wn),ee(this,Vr),ee(this,Ss,[0,0,0,0]),ee(this,Un,[]),ee(this,Ks),ee(this,Xi),ee(this,Yi),ee(this,Zi,[0,0,0,0]),ee(this,qr),ee(this,zn),!n&&ue(ie.CANVAS_NOT_FOUND);const i=n.getContext("webgpu");!i&&ue(ie.CONTEXT_NOT_FOUND),i.configure({device:e,...r}),j(this,Qr,this.CreateBuffer({size:T(this,uo).length*Float32Array.BYTES_PER_ELEMENT,label:"Render Pipeline Resolution Buffer",usage:Ct.UNIFORM})),j(this,Ge,n),j(this,js,i),se(this,ls,uc).call(this),j(this,Vr,r.format),this.CreatePassDescriptor(this.CreateColorAttachment())}ConfigureContext(e){const t=e.format??T(this,Vr);T(this,js).configure({device:this.Device,format:t,...e})}CreateColorAttachment(e,t="clear",n="store",r,i,o){return{view:e,loadOp:t,storeOp:n,clearValue:r&&se(this,ls,cc).call(this,r),resolveTarget:i,depthSlice:o}}CreateDepthAttachment(e,t=1,n="clear",r="store",i){return j(this,Hs,!0),j(this,Jr,new Ch(this.Device)),{view:e,depthClearValue:t,depthLoadOp:n,depthStoreOp:r,depthReadOnly:i}}CreateStencilAttachment(e,t="clear",n="store",r){return{stencilClearValue:e,stencilLoadOp:t,stencilStoreOp:n,stencilReadOnly:r}}CreatePassDescriptor(e,t,n,r,i,o){const a=Array.isArray(e)&&e||[e];return j(this,as,!a.some(({view:l})=>!!l)),t??(t=this.CreatePipelineLabel("Render Pass")),this.Descriptor={colorAttachments:a,depthStencilAttachment:n,occlusionQuerySet:r,timestampWrites:i,maxDrawCount:o,label:t}}CreateIndexBuffer(e,t){const n=t?.label??"Index Buffer";return e=Array.isArray(e)&&new Uint32Array(e)||e,this.CreateBuffer({label:n,size:e.byteLength,usage:Ct.INDEX,...t})}CreateVertexBufferAttribute(e,t=0,n=0){return se(this,ls,hc).call(this,e,t,n)}CreateVertexBufferLayout(e,t,n="vertex"){!this.Reflect&&ue(ie.SHADER_MODULE_NOT_FOUND,"`LegacyRenderer.CreateVertexBufferLayout`.\n            Call `LegacyRenderer.CreateShaderModule` before creating a vertex layout or vertex buffer.");const{entry:{vertex:r}}=this.Reflect,i=r.find(({name:l})=>n===l);!i&&ue(ie.VERTEX_ENTRY_NOT_FOUND,`\`${n}\` in vertex shader entries.`);let o=[],a=0;for(let l=0,u=(e=Array.isArray(e)&&e||[e]).length;l<u;++l){const c=e[l],h=typeof c=="string",d=h?c:c.name,w=i.inputs.find(({name:k})=>d===k);if(w){const k=h?Yy(w.type.size):c.format;o.push(se(this,ls,hc).call(this,k,+w.location,a)),a+=Zy(k)}else Bt(ie.VERTEX_ATTRIBUTE_NOT_FOUND,`\`${d}\` in vertex shader inputs.`)}return{arrayStride:a,stepMode:t,attributes:o}}CreateVertexBuffer(e,t=1,n,r="vertex"){const i=t.label??"Vertex Buffer";if(e instanceof Float32Array)return this.CreateBuffer({label:i,size:e.byteLength,usage:Ct.VERTEX,...t});const o=this.CreateVertexBufferLayout(e,n,r),a=(typeof t=="number"&&t||(t.count??1))*o.arrayStride;return{buffer:this.CreateBuffer({label:i,size:a,usage:Ct.VERTEX,...t}),layout:o}}CreateVertexState(e,t="vertex",n,r){return{module:e,entryPoint:t,buffers:n=Array.isArray(n)&&n||[n],constants:r}}CreateBlendComponent(e="add",t="one",n="zero"){return{operation:e,srcFactor:t,dstFactor:n}}CreateTargetState(e=T(this,Vr),t,n){return t&&(t={color:t.color??{},alpha:t.alpha??{}}),{format:e,blend:t,writeMask:n}}CreateFragmentState(e,t="fragment",n,r){return n??(n=this.CreateTargetState()),{module:e,entryPoint:t,targets:n=Array.isArray(n)&&n||[n],constants:r}}CreateStencilFaceState(e,t,n,r){return{compare:e,failOp:t,depthFailOp:n,passOp:r}}CreateDepthStencilState(e="depth24plus",t=!0,n="less",r,i,o,a,l,u,c){return{format:e,depthWriteEnabled:t,depthCompare:n,stencilFront:r,stencilBack:i,stencilReadMask:o,stencilWriteMask:a,depthBias:l,depthBiasSlopeScale:u,depthBiasClamp:c}}CreateMultisampleState(e=4,t,n){return{count:e,mask:t,alphaToCoverageEnabled:n}}CreateStorageTextureBindingLayout(e=T(this,Vr),t,n,r,i){return{binding:i,visibility:GPUShaderStage.FRAGMENT,storageTexture:{access:t,format:e,viewDimension:n}}}CreatePipeline(e={},t){let n=this.GetShaderModule(e),{vertex:r,fragment:i}=e;!n&&!r&&(n=this.CreateShaderModule()),n&&(r??(r=this.CreateVertexState(n)),i??(i=this.CreateFragmentState(n)));const o=e.label??this.CreatePipelineLabel("Render Pipeline"),a=e.layout??"auto";return this.SetPipeline(this.Device.createRenderPipeline({label:o,layout:a,vertex:r,fragment:i,...e})),t&&(T(this,at)?T(this,at).setPipeline(this.Pipeline):Bt(ie.RENDER_PASS_NOT_FOUND)),this.Pipeline}SavePipelineState(){super.SavePipelineState(),j(this,zn,T(this,wn)),j(this,Ks,T(this,Un)),j(this,Zi,T(this,Ss)),j(this,qr,T(this,_n)),j(this,Xi,T(this,as)),j(this,Yi,T(this,Hs)),T(this,zn)&&j(this,zn,Object.values(T(this,zn)))}ResetPipelineState(){var e;super.ResetPipelineState(),this.SetVertexBuffers([]),this.SetIndexBuffer(void 0),j(this,as,!1),j(this,Zi,[0,0,0,0]),j(this,Hs,!1),j(this,_n,(e=T(this,_n))==null?void 0:e.destroy())}RestorePipelineState(){super.RestorePipelineState(),j(this,Un,T(this,Ks)),j(this,Ss,T(this,Zi)),j(this,_n,T(this,qr)),j(this,as,T(this,Xi)),j(this,Hs,T(this,Yi)),this.SetIndexBuffer(...Array.isArray(T(this,zn))&&T(this,zn)||[void 0])}SetVertexBuffers(e,t,n){t=Array.isArray(t)&&t||[t],n=Array.isArray(n)&&n||[n],j(this,Un,Array.isArray(e)&&e.map((r,i)=>({buffer:r,offset:t[i],size:n[i]}))||[{buffer:e,offset:t[0],size:n[0]}])}AddVertexBuffers(e,t,n){t=Array.isArray(t)&&t||[t],n=Array.isArray(n)&&n||[n],T(this,Un).push(...Array.isArray(e)&&e.map((r,i)=>({buffer:r,offset:t[i],size:n[i]}))||[{buffer:e,offset:t[0],size:n[0]}])}SetIndexBuffer(e,t="uint32",n,r){j(this,wn,e&&{buffer:e,format:t,offset:n,size:r})}SetCanvasSize(e,t,n=!0){!this.Device&&ue(ie.DEVICE_NOT_FOUND),!T(this,Ge)&&ue(ie.CANVAS_NOT_FOUND);let r=this.DevicePixelRatio*e|0,i=this.DevicePixelRatio*t|0;const o=this.Device.limits.maxTextureDimension2D;r=Math.max(1,Math.min(r,o)),i=Math.max(1,Math.min(i,o)),T(this,Ge).width===r&&T(this,Ge).height===i||(T(this,Ge).height=i,T(this,Ge).width=r,se(this,ls,uc).call(this),n&&(T(this,Ge).style.width=e+"px",T(this,Ge).style.height=t+"px"))}SetTextureView(e,t=0){this.Descriptor.colorAttachments[t].view=e,j(this,as,!e)}UpdateOrthographicProjection(e=1,t=1e3,n=0,r=T(this,Ge).clientWidth,i=T(this,Ge).clientHeight,o=0){return Gd.ortho(o,r,i,n,e,t,T(this,Ki)),T(this,Ki)}UpdateProjection2D(e=T(this,Ge).clientWidth,t=T(this,Ge).clientHeight){return vl.set(2/e,0,0,0,-2/t,0,-1,1,1,T(this,ji)),T(this,ji)}Render(e,t=!0){T(this,Hs)&&se(this,ls,qy).call(this),T(this,at)||(T(this,_n)?(this.Descriptor.colorAttachments[0].view=T(this,_n).createView(),this.Descriptor.colorAttachments[0].resolveTarget=this.CurrentTextureView):T(this,as)&&(this.Descriptor.colorAttachments[0].view=this.CurrentTextureView),j(this,at,this.GetCommandEncoder().beginRenderPass(this.Descriptor)),T(this,at).setPipeline(this.Pipeline),j(this,$a,T(this,wn)?T(this,at).drawIndexed.bind(T(this,at)):T(this,at).draw.bind(T(this,at))));for(let n=0,r=T(this,Un).length;n<r;++n){const{buffer:i,offset:o,size:a}=T(this,Un)[n];T(this,at).setVertexBuffer(n,i,o,a)}T(this,wn)&&T(this,at).setIndexBuffer(T(this,wn).buffer,T(this,wn).format,T(this,wn).offset,T(this,wn).size);for(let n=0,r=0,i=this.BindGroups.length;n<i;++n){const{bindGroup:o,dynamicOffsets:a,active:l}=this.BindGroups[n];l&&T(this,at).setBindGroup(r++,o,a)}T(this,at).setBlendConstant(T(this,Ss)),T(this,$a).call(this,...Array.isArray(e)&&e||[e]),t&&this.Submit()}DestroyCurrentPass(){var e;(e=T(this,at))==null||e.end(),j(this,at,void 0)}Submit(){this.DestroyCurrentPass(),this.SubmitCommandBuffer(),this.SetCommandEncoder(void 0)}get Canvas(){return T(this,Ge)}get Context(){return T(this,js)}get CurrentPass(){return T(this,at)}get AspectRatio(){return!T(this,Ge)&&ue(ie.CANVAS_NOT_FOUND),T(this,Ge).width/T(this,Ge).height}get Projection2D(){return T(this,ji)}get DepthTexture(){return T(this,vn)}get CurrentTexture(){return T(this,js).getCurrentTexture()}get CurrentTextureView(){return this.CurrentTexture.createView()}get OrthographicProjection(){return T(this,Ki)}set MultisampleTexture(e){j(this,_n,e)}get MultisampleTexture(){return T(this,_n)}set BlendConstant(e){j(this,Ss,se(this,ls,cc).call(this,e))}get BlendConstant(){return T(this,Ss)}get ResolutionBuffer(){return T(this,Qr)}get DevicePixelRatio(){return globalThis.devicePixelRatio??1}set TextureView(e){this.Descriptor.colorAttachments[0].view=e,j(this,as,!e)}get BaseCanvasSize(){const{width:e,height:t}=T(this,Ge),n=this.DevicePixelRatio;return[e/n,t/n]}get CanvasSize(){return[T(this,Ge).width,T(this,Ge).height]}Destroy(){var e,t,n,r,i,o,a;super.Destroy(),this.DestroyCurrentPass(),T(this,Qr).destroy(),j(this,Ss,[0,0,0,0]),j(this,Jr,(e=T(this,Jr))==null?void 0:e.Destroy()),j(this,vn,(t=T(this,vn))==null?void 0:t.destroy()),j(this,qr,(n=T(this,qr))==null?void 0:n.destroy()),j(this,Xi,j(this,Yi,!1)),(r=T(this,Ks))==null||r.forEach(({buffer:l})=>l.destroy()),T(this,Un).forEach(({buffer:l})=>l.destroy()),(i=T(this,wn))==null||i.buffer.destroy(),(o=T(this,zn))==null||o.splice(0),j(this,zn,void 0),(a=T(this,Ks))==null||a.splice(0),j(this,Ks,void 0),T(this,Un).splice(0),this.ResetPipelineState(),T(this,js).unconfigure()}}$a=new WeakMap,as=new WeakMap,ji=new WeakMap,uo=new WeakMap,Hs=new WeakMap,Ge=new WeakMap,js=new WeakMap,Ki=new WeakMap,Qr=new WeakMap,Jr=new WeakMap,vn=new WeakMap,_n=new WeakMap,at=new WeakMap,wn=new WeakMap,Vr=new WeakMap,Ss=new WeakMap,Un=new WeakMap,Ks=new WeakMap,Xi=new WeakMap,Yi=new WeakMap,Zi=new WeakMap,qr=new WeakMap,zn=new WeakMap,ls=new WeakSet,uc=function(){T(this,uo).set([T(this,Ge).width,T(this,Ge).height,this.DevicePixelRatio]),this.WriteBuffer(T(this,Qr),T(this,uo))},cc=function(s){return s instanceof gl?s.rgba:s},hc=function(s,e=0,t=0){return{format:s,shaderLocation:e,offset:t}},qy=function(){var s,e;const t=this.CurrentTexture,{width:n,height:r}=t;T(this,vn)&&T(this,vn).width===n&&T(this,vn).height===r||((s=T(this,vn))==null||s.destroy(),j(this,vn,T(this,Jr).CreateTextureFromSource(t,{sampleCount:((e=T(this,_n))==null?void 0:e.sampleCount)??1,usage:GPUTextureUsage.RENDER_ATTACHMENT,label:"Depth Texture",format:"depth24plus",mipmaps:!1}))),this.Descriptor.depthStencilAttachment.view=T(this,vn).createView()};var Da;class iC extends zy{constructor(e,t){super(e,t,"Compute"),ee(this,Da,[1]),this.CreatePassDescriptor()}CreatePassDescriptor(e,t){return e??(e=this.CreatePipelineLabel("Compute Pass")),this.Descriptor={label:e,timestampWrites:t}}CreatePipeline(e){const t=e.label??this.CreatePipelineLabel("Compute Pipeline"),n=e.layout??"auto",r=this.GetShaderModule(e)??this.CreateShaderModule();return this.SetPipeline(this.Device.createComputePipeline({label:t,layout:n,compute:{module:r,...e}}))}Compute(e=!1){const t=this.GetCommandEncoder().beginComputePass(this.Descriptor);t.setPipeline(this.Pipeline);for(let n=0,r=0,i=this.BindGroups.length;n<i;++n){const{bindGroup:o,dynamicOffsets:a,active:l}=this.BindGroups[n];l&&t.setBindGroup(r++,o,a)}t.dispatchWorkgroups(...T(this,Da)),t.end(),e&&this.Submit()}Submit(){this.SubmitCommandBuffer(),this.SetCommandEncoder(void 0)}set Workgroups(e){j(this,Da,Array.isArray(e)&&e||[e])}Destroy(){super.Destroy(),this.Workgroups=1}}Da=new WeakMap;var Xs,Ys,sr,Hr,Zs,jr;class Hy{constructor(e,t,n){ee(this,Zs),Lt(this,"Pipelines",[]),ee(this,Xs),Lt(this,"Device"),ee(this,Ys),ee(this,sr),ee(this,Hr),j(this,Hr,n),j(this,sr,t),this.Device=e,j(this,Ys,this.CreateStageLabel("Command Encoder"))}CreateStageLabel(e){return T(this,Hr)&&e&&`${T(this,Hr)} ${e}`||""}CreateTimestampWrites(e,t=0,n=1){return{querySet:e,beginningOfPassWriteIndex:t,endOfPassWriteIndex:n}}ResolveQuerySet(e,t,n=0,r=e.count,i=0){this.GetCommandEncoder(!0).resolveQuerySet(e,n,r,t,i)}CreateBufferBindingLayout(e,t,n,r,i){return r??(r=se(this,Zs,jr).call(this)),{binding:i,visibility:r,buffer:{type:e,hasDynamicOffset:t,minBindingSize:n}}}CreateSamplerBindingLayout(e,t,n){return t??(t=se(this,Zs,jr).call(this)),{binding:n,visibility:t,sampler:{type:e}}}CreateTextureBindingLayout(e,t,n,r,i){return r??(r=se(this,Zs,jr).call(this)),{binding:i,visibility:r,texture:{sampleType:e,viewDimension:t,multisampled:n}}}CreateStorageTextureBindingLayout(e,t,n,r,i){return r??(r=se(this,Zs,jr).call(this)),{binding:i,visibility:r,storageTexture:{access:t,format:e,viewDimension:n}}}CreateExternalTextureBindingLayout(e,t){return e??(e=se(this,Zs,jr).call(this)),{binding:t,visibility:e,externalTexture:{}}}CreateCommandEncoder(){return j(this,Xs,this.Device.createCommandEncoder({label:T(this,Ys)}))}GetCommandEncoder(e=!1){if(!T(this,Xs)){if(e){const t=""+(T(this,Ys)&&`Label: "${T(this,Ys)}".`);Bt(ie.COMMAND_ENCODER_NOT_FOUND,` ${t} Creating a new one.`)}return this.CreateCommandEncoder()}return T(this,Xs)}SubmitCommandBuffer(){this.Device.queue.submit([T(this,Xs).finish()])}CreateBuffer(e){const t=e.label??this.CreateStageLabel("Buffer");return this.Device.createBuffer({label:t,...e})}WriteBuffer(e,t,n=0,r,i){this.Device.queue.writeBuffer(e,n,t,r,i)}CopyBufferToBuffer(e,t,n=t.size,r=0,i=0){this.GetCommandEncoder(!0).copyBufferToBuffer(e,r,t,i,n)}RemovePipeline(e){const t=this.Pipelines.indexOf(e);t<0?(Bt(ie.PIPELINE_NOT_FOUND,`${T(this,sr)}Pipeline. The following pipeline was not found when
                calling \`${T(this,sr)==="Render"&&T(this,sr)+"er"||"Computation"}.RemovePipeline\` method.`),console.warn(e)):(this.Pipelines[t].Destroy(),this.Pipelines.splice(t,1))}set CommandEncoder(e){j(this,Xs,e)}set CommandEncoderLabel(e){j(this,Ys,e)}get Name(){return T(this,Hr)}Destroy(){this.Pipelines.forEach(e=>e.Destroy()),this.CommandEncoder=void 0,this.Pipelines.splice(0)}}Xs=new WeakMap,Ys=new WeakMap,sr=new WeakMap,Hr=new WeakMap,Zs=new WeakSet,jr=function(){return T(this,sr)==="Render"&&GPUShaderStage.FRAGMENT||GPUShaderStage.COMPUTE};var Sl,Oa,Ma,fc;class oC extends Hy{constructor(e,t){super(e,"Compute",t),ee(this,Ma),ee(this,Sl,[1]),ee(this,Oa),this.CreatePassDescriptor()}CreatePassDescriptor(e,t){return e??(e=this.CreateStageLabel("Compute Pass")),j(this,Oa,{label:e,timestampWrites:t})}GetMaxEvenWorkgroupDimension(e=1){const{maxComputeInvocationsPerWorkgroup:t}=this.Device.limits;return 0|(e===3?Math.cbrt(t):e===2?Math.sqrt(t):t)}async CreatePipeline(e){const t=new this.Pipeline(e.pipelineName);return e=jl(e)??t.CreateShaderModule(...Object.values((Array.isArray(e)||typeof e=="string")&&[e]||e)),await this.AddPipeline(t,e),t}async AddPipeline(e,t){return await e.Init(t),Reflect.deleteProperty(e,"Init"),this.Pipelines.push(e),e}Compute(e=!0){const t=this.GetCommandEncoder().beginComputePass(T(this,Oa)),n=this.Pipelines.length;if(n-1||!this.Pipelines[0].Active)for(let r=0;r<n;++r){const i=this.Pipelines[r];i.Active&&se(this,Ma,fc).call(this,t,i)}else se(this,Ma,fc).call(this,t,this.Pipelines[0]);e&&this.Submit()}Submit(){this.SubmitCommandBuffer(),this.CommandEncoder=void 0}set Workgroups(e){j(this,Sl,lt(e).map(Math.ceil))}get Pipeline(){const{Name:e,Device:t}=this;return class extends YA{constructor(n=e){super(t,n)}}}Destroy(){super.Destroy(),this.Workgroups=1}}Sl=new WeakMap,Oa=new WeakMap,Ma=new WeakSet,fc=function(s,e){s.setPipeline(e.GPUPipeline),e.UseBindGroups(s),s.dispatchWorkgroups(...T(this,Sl)),s.end()};var Pa,co,Ra,Je,Qs,ei,ti,La,ni,Sn,Kr,ur,Et,Js,jy,dc,pc;class aC extends Hy{constructor(e,t,n,r){super(e,"Render",t),ee(this,Js),ee(this,Pa),ee(this,co,new Float32Array(3)),ee(this,Ra,!1),ee(this,Je),ee(this,Qs),ee(this,ei),ee(this,ti),ee(this,La),ee(this,ni),ee(this,Sn),ee(this,Kr),ee(this,ur),ee(this,Et),!n&&ue(ie.CANVAS_NOT_FOUND);const i=n.getContext("webgpu");!i&&ue(ie.CONTEXT_NOT_FOUND),i.configure({device:e,...r}),j(this,ei,this.CreateBuffer({size:T(this,co).length*Float32Array.BYTES_PER_ELEMENT,label:"Render Pipeline Resolution Buffer",usage:Ct.UNIFORM})),j(this,Je,n),j(this,Qs,i),se(this,Js,dc).call(this),j(this,Kr,r.format),this.CreatePassDescriptor(this.CreateColorAttachment())}ConfigureContext(e){const t=e.format??T(this,Kr);T(this,Qs).configure({device:this.Device,format:t,...e})}CreateColorAttachment(e,t,n="clear",r="store",i,o){return{view:t,loadOp:n,storeOp:r,clearValue:e&&t0(e),resolveTarget:i,depthSlice:o}}CreateDepthStencilAttachment(e,t=1,n="clear",r="store",i,o,a="clear",l="store",u){return j(this,Ra,!0),j(this,ti,new Ch(this.Device)),{view:e,depthClearValue:t,depthLoadOp:n,depthStoreOp:r,depthReadOnly:i,stencilClearValue:o,stencilLoadOp:a,stencilStoreOp:l,stencilReadOnly:u}}CreatePassDescriptor(e,t,n,r,i,o){const a=lt(e);return n??(n=this.CreateStageLabel("Render Pass")),j(this,ni,{colorAttachments:a,depthStencilAttachment:t,occlusionQuerySet:r,timestampWrites:i,maxDrawCount:o,label:n})}CreateStorageTextureBindingLayout(e=T(this,Kr),t,n,r,i){return{binding:i,visibility:GPUShaderStage.FRAGMENT,storageTexture:{access:t,format:e,viewDimension:n}}}SetCanvasSize(e,t,n=!0){!this.Device&&ue(ie.DEVICE_NOT_FOUND),!T(this,Je)&&ue(ie.CANVAS_NOT_FOUND);let r=this.DevicePixelRatio*e|0,i=this.DevicePixelRatio*t|0;const o=this.Device.limits.maxTextureDimension2D;r=Math.max(1,Math.min(r,o)),i=Math.max(1,Math.min(i,o)),T(this,Je).width===r&&T(this,Je).height===i||(T(this,Je).height=i,T(this,Je).width=r,se(this,Js,dc).call(this),n&&(T(this,Je).style.width=e+"px",T(this,Je).style.height=t+"px"))}async CreatePipeline(e,t){const n=new this.Pipeline(e.pipelineName);return e=jl(e)??n.CreateShaderModule(...Object.values((Array.isArray(e)||typeof e=="string")&&[e]||e)),await this.AddPipeline(n,e,t),n}async AddPipeline(e,t,n){const r=await e.Init(t);return n&&(T(this,Et)?T(this,Et).setPipeline(r):Bt(ie.RENDER_PASS_NOT_FOUND)),Reflect.deleteProperty(e,"Init"),this.Pipelines.push(e),e}Render(e=!0){T(this,Ra)&&se(this,Js,jy).call(this);const t=this.Pipelines.length;if(t-1||!this.Pipelines[0].Active)for(let n=0;n<t;++n){const r=this.Pipelines[n];r.Active&&se(this,Js,pc).call(this,r,e)}else se(this,Js,pc).call(this,this.Pipelines[0],e);e&&this.Submit()}DestroyRenderPass(){var e;(e=T(this,Et))==null||e.end(),j(this,Et,void 0)}Submit(){this.DestroyRenderPass(),this.SubmitCommandBuffer(),this.CommandEncoder=void 0}get Canvas(){return T(this,Je)}get Context(){return T(this,Qs)}get RenderPass(){return T(this,Et)}get DepthTexture(){return T(this,Sn)}get CurrentTexture(){return T(this,Qs).getCurrentTexture()}get CurrentTextureView(){return this.CurrentTexture.createView()}set MultisampleTexture(e){j(this,ur,e)}get MultisampleTexture(){return T(this,ur)}set DevicePixelRatio(e){j(this,La,e)}get DevicePixelRatio(){return T(this,La)??globalThis.devicePixelRatio??1}get ResolutionBuffer(){return T(this,ei)}get BaseCanvasSize(){const{width:e,height:t}=T(this,Je),n=this.DevicePixelRatio;return[e/n,t/n]}get CanvasSize(){return[T(this,Je).width,T(this,Je).height]}get AspectRatio(){return!T(this,Je)&&ue(ie.CANVAS_NOT_FOUND),T(this,Je).width/T(this,Je).height}get Pipeline(){const{Name:e,Device:t}=this,n=T(this,Kr);return class extends XA{constructor(r=e){super(t,n,r)}}}Destroy(){var e,t;super.Destroy(),this.DestroyRenderPass(),T(this,ei).destroy(),j(this,Sn,(e=T(this,Sn))==null?void 0:e.destroy()),j(this,ti,(t=T(this,ti))==null?void 0:t.Destroy()),T(this,Qs).unconfigure()}}Pa=new WeakMap,co=new WeakMap,Ra=new WeakMap,Je=new WeakMap,Qs=new WeakMap,ei=new WeakMap,ti=new WeakMap,La=new WeakMap,ni=new WeakMap,Sn=new WeakMap,Kr=new WeakMap,ur=new WeakMap,Et=new WeakMap,Js=new WeakSet,jy=function(){var s,e;const t=this.CurrentTexture,{width:n,height:r}=t;T(this,Sn)&&T(this,Sn).width===n&&T(this,Sn).height===r||((s=T(this,Sn))==null||s.destroy(),j(this,Sn,T(this,ti).CreateTextureFromSource(t,{sampleCount:((e=T(this,ur))==null?void 0:e.sampleCount)??1,usage:GPUTextureUsage.RENDER_ATTACHMENT,label:"Depth Texture",format:"depth24plus",mipmaps:!1}))),T(this,ni).depthStencilAttachment.view=T(this,Sn).createView()},dc=function(){T(this,co).set([T(this,Je).width,T(this,Je).height,this.DevicePixelRatio]),this.WriteBuffer(T(this,ei),T(this,co))},pc=function(s,e){if(!T(this,Et)){let t="view";const n=T(this,ni).colorAttachments[s.ColorAttachment];T(this,ur)&&(t="resolveTarget",n.view=T(this,ur).createView()),n[t]=s.TextureView||this.CurrentTextureView,j(this,Et,this.GetCommandEncoder().beginRenderPass(T(this,ni))),j(this,Pa,T(this,Et)[s.DrawMethod].bind(T(this,Et))),T(this,Et).setPipeline(s.GPUPipeline)}s.UseRenderBuffers(T(this,Et)),s.UseBindGroups(T(this,Et)),T(this,Et).setBlendConstant(s.BlendConstant),T(this,Pa).call(this,...s.DrawParams),s.DestroyPassEncoder&&!e&&this.DestroyRenderPass()};var ho,ai,fo,po,gs,fs,Qi,Vd,Ky,Xy;const Wn=class{static async CreateQuerySet(s,e){const t=(await this.GPUDevice).createQuerySet({type:s,count:e});return T(this,ho).push(t),t}static RenderPipeline(s,e="",t={}){return t.format??(t.format=this.PreferredCanvasFormat),se(this,fs,Qi).call(this,e),(async()=>{const n=await this.GPUDevice;return new Proxy(rC,{construct(r){return new r(n,e,s,t)}})})()}static Renderer(s,e="",t={}){return t.format??(t.format=this.PreferredCanvasFormat),se(this,fs,Qi).call(this,e),(async()=>{const n=await this.GPUDevice;return new Proxy(aC,{construct(r){return new r(n,e,s,t)}})})()}static ComputePipeline(s=""){return se(this,fs,Qi).call(this,s),(async()=>{const e=await this.GPUDevice;return new Proxy(iC,{construct(t){return new t(e,s)}})})()}static Computation(s=""){return se(this,fs,Qi).call(this,s),(async()=>{const e=await this.GPUDevice;return new Proxy(oC,{construct(t){return new t(e,s)}})})()}static LegacyTexture(s){return(async()=>{const e=await this.GPUDevice;return new Proxy(Md,{construct(t){return new Md(e,s)}})})()}static Texture(s){return(async()=>{const e=await this.GPUDevice;return new Proxy(Ch,{construct(t){return new t(e,s)}})})()}static Destroy(s,e){var t;T(this,ho).forEach(n=>n.destroy()),(s=lt(s)).forEach(n=>n?.destroy()),(e=lt(e)).forEach(n=>n?.destroy()),(t=T(this,ai))==null||t.destroy(),T(this,ho).splice(0),T(this,gs).requiredFeatures.clear(),j(this,fo,j(this,ai,null)),this.DescriptorLabel=this.RequiredLimits=void 0,this.PowerPreference=this.ForceFallbackAdapter=void 0}static set PowerPreference(s){T(this,po).powerPreference=s}static set ForceFallbackAdapter(s){T(this,po).forceFallbackAdapter=s}static set DescriptorLabel(s){T(this,gs).label=s}static async SetRequiredFeatures(s){const e=(await this.Adapter).features;return(s=lt(s)).forEach(t=>e.has(t)?T(this,gs).requiredFeatures.add(t):Bt(ie.FEATURE_NOT_FOUND,`"${t}".
It will be skipped when requesting a GPUDevice.`)),T(this,gs).requiredFeatures}static set RequiredLimits(s){T(this,gs).requiredLimits=s}static get PreferredCanvasFormat(){return!navigator.gpu&&ue(ie.WEBGPU_NOT_SUPPORTED),navigator.gpu.getPreferredCanvasFormat()}static get Adapter(){return(async()=>T(this,fo)??await se(this,fs,Ky).call(this)())()}static get GPUDevice(){return(async()=>T(this,ai)??await se(this,fs,Xy).call(this)())()}static set OnDeviceLost(s){this.OnLost=s}static get OnDeviceLost(){return this.OnLost}static get Device(){return this.GPUDevice}static get VERSION(){return"0.0.12"}};ho=new WeakMap,ai=new WeakMap,fo=new WeakMap,po=new WeakMap,gs=new WeakMap,fs=new WeakSet,Qi=function(s){var e;(e=T(this,gs)).label??(e.label=s&&s+" Device"||"")},Vd=function(s){if(Wn.OnLost)return Wn.OnLost(s);const e=(s.message&&" | Message: "+s.message)??".";ue(ie.DEVICE_LOST,"Reason: "+s.reason+e)},Ky=function(){return!navigator.gpu&&ue(ie.WEBGPU_NOT_SUPPORTED),async()=>{const s=await navigator.gpu.requestAdapter(T(this,po));return!s&&ue(ie.ADAPTER_NOT_FOUND),j(this,fo,s)}},Xy=function(){return async()=>{const{requiredFeatures:s,requiredLimits:e,label:t}=T(this,gs),n=await(await this.Adapter).requestDevice({requiredFeatures:s,requiredLimits:e,defaultQueue:{label:t}});return!n&&ue(ie.DEVICE_NOT_FOUND),n.lost.then(se(this,fs,Vd)),j(this,ai,n)}},ee(Wn,fs),ee(Wn,ho,[]),ee(Wn,ai,null),ee(Wn,fo,null),ee(Wn,po,{powerPreference:void 0,forceFallbackAdapter:!1}),Lt(Wn,"OnLost"),ee(Wn,gs,{label:void 0,requiredFeatures:new Set,requiredLimits:void 0});let En=Wn;function Ue(s){for(let e in s)s[e]={value:s[e]};return Object.freeze(Object.create(null,s))}function Yy(s){switch(s){case 2:return"unorm8x2";case 4:return"float32";case 8:return"float32x2";case 12:return"float32x3";case 16:return"float32x4"}}function Zy(s){switch(s){case"uint8x2":case"sint8x2":case"unorm8x2":case"snorm8x2":return 2;case"uint32":case"sint32":case"float32":case"uint8x4":case"sint8x4":case"unorm8x4":case"snorm8x4":case"uint16x2":case"sint16x2":case"unorm16x2":case"snorm16x2":case"float16x2":return 4;case"uint16x4":case"sint16x4":case"uint32x2":case"sint32x2":case"unorm16x4":case"snorm16x4":case"float16x4":case"float32x2":return 8;case"uint32x3":case"sint32x3":case"float32x3":return 12;case"uint32x4":case"sint32x4":case"float32x4":return 16}return 0}function Qy(s){return s==="f16"||s.includes("h")?"f16":s.includes("f")?"f32":s.includes("u")?"u32":"i32"}function Jy(s){return+s.slice(1)/8}function e0(s){return s==="f16"&&ue(ie.FORMAT_NOT_SUPPORTED,s+"."),s==="f32"?Float32Array:s==="u32"?Uint32Array:Int32Array}function lt(s){return Array.isArray(s)&&s||[s]}function t0(s){return s instanceof gl?s.rgba:s}function jl(s){return s instanceof GPUShaderModule&&s||s.module}Ue({TRIANGLE:3,SQUARE:4,PENTAGON:5,HEXAGON:6,HEPTAGON:7,OCTAGON:8,NONAGON:9,DECAGON:10,DODECAGON:12});var cr,si,ri,rr,ii,kl,Il,mo,go,Ts,Es,hr,yo,Ke,Tn,Ji,er,Fe,kn,n0,s0,qd,Hd,El,mc;const _u=class eo{constructor(e){ee(this,kn),ee(this,si),ee(this,ri),ee(this,rr),ee(this,ii),ee(this,kl),ee(this,Il),ee(this,mo),ee(this,go),ee(this,Ts),ee(this,Es),ee(this,hr),ee(this,yo),ee(this,Ke),ee(this,Tn,[]),ee(this,Ji,vl.create()),ee(this,er,[void 0]),ee(this,Fe,Ue({min:xu.create(),max:xu.create(),size:xu.create()}));const{font:t,color:n,renderer:r,size:i=16,background:o,lineGap:a=0,hinting:l=!0,subpixel:u=!0,label:c="SDFText"}=e;j(this,si,t),j(eo,cr,c),j(this,kl,l),j(this,Il,u),j(this,Ke,r),this.Size=i??16,se(this,kn,n0).call(this),this.Color=n??[0,0,0,1],j(this,ii,T(this,rr)*a),this.Background=o??[1,1,1,1]}static async GetFragmentStateParams(e,t="",n,r,i){return eo.GetShaderStateParams(e,t,n,r,i)}static async GetShaderStateParams(e,t="",n,r,i){const o=e.CreateBlendComponent(n,r??"src1",i??"one-minus-src1");let a=Array.isArray(t)&&t.join(`

`)||t;const l=(await En.GPUDevice).features.has("dual-source-blending");a=`${Gy}

${a}`,l&&(a=`${Wy}

${a}`);const u=e.CreateShaderModule(a,T(eo,cr)+" Shader Module"),c=e.CreateVertexBufferLayout(["position","texture","size"],void 0,"textVertex");return{target:e.CreateTargetState(void 0,l&&{color:o}||void 0),fragmentEntry:l?"dsbTextFragment":"textFragment",constants:{TRIPLET_FACTOR:.6},vertexEntry:"textVertex",module:u,layout:c,shader:a}}async SetFontTexture(e,t="r8unorm"){const n=new(await En.LegacyTexture()),r=n.CopyImageToTexture(e,{create:{format:t},mipmaps:!1});T(this,yo).set([r.width,r.height]),T(this,Ke).WriteBuffer(T(this,Ts),T(this,yo)),T(this,er)[0]=T(this,Ke).CreateBindGroup(T(this,Ke).CreateBindGroupEntries([n.CreateSampler({filter:"linear"}),{buffer:T(this,Ts)},{buffer:T(this,Es)},r.createView()]),0,T(eo,cr)+" Bind Group")}AddVertexBuffers(e){T(this,Tn).push(...lt(bindGroups))}AddBindGroups(e){T(this,er).push(...lt(e))}Write(e,t=[0,0]){!T(this,er)[0]&&ue(ie.FONT_TEXTURE_NOT_FOUND,"`SDFText.Write` method. Call `SDFText.SetFontTexture` method before writing any string."),j(this,ri,e),se(this,kn,s0).call(this),t=t.map(n=>-n),se(this,kn,mc).call(this,T(this,hr),t)}Render(e=!0){T(this,Ke).SavePipelineState(),T(this,Ke).SetBindGroups(T(this,er)),T(this,Ke).SetVertexBuffers(T(this,Tn)),T(this,Ke).Render(T(this,hr).length/5,e),T(this,Ke).RestorePipelineState()}Resize(){se(this,kn,El).call(this)}Destroy(){T(this,Tn).forEach(e=>e.destroy()),j(this,Ts,T(this,Ts).destroy()),j(this,Es,T(this,Es).destroy()),T(this,Tn).splice(0),T(this,er).splice(1)}set Position([e,t]){T(this,Fe).min[0]=e,T(this,Fe).min[1]=t,T(this,Fe).size[0]=T(this,Fe).max[0]-e,T(this,Fe).size[1]=T(this,Fe).max[1]-t,T(this,Fe).max[0]+=e,T(this,Fe).max[1]+=t,se(this,kn,El).call(this)}set Background(e){T(this,mo).set(e instanceof gl?e.rgba:e),T(this,Ke).WriteBuffer(T(this,Es),T(this,mo).buffer)}set Color(e){T(this,go).set(e instanceof gl?e.rgba:e),T(this,Ke).WriteBuffer(T(this,Es),T(this,go).buffer)}set Size(e){const t=T(this,ii)/T(this,rr);if(j(this,rr,Math.round(T(this,Ke).DevicePixelRatio*e)),!T(this,Tn).length)return;j(this,ii,T(this,rr)*t);const n=T(this,Fe).min.map((r,i)=>r*(-1*i+.5));se(this,kn,mc).call(this,T(this,hr),n)}get BoundingBox(){return T(this,Fe)}};cr=new WeakMap,si=new WeakMap,ri=new WeakMap,rr=new WeakMap,ii=new WeakMap,kl=new WeakMap,Il=new WeakMap,mo=new WeakMap,go=new WeakMap,Ts=new WeakMap,Es=new WeakMap,hr=new WeakMap,yo=new WeakMap,Ke=new WeakMap,Tn=new WeakMap,Ji=new WeakMap,er=new WeakMap,Fe=new WeakMap,kn=new WeakSet,n0=function(){const{buffer:s,Text:{matrix:e,textureSize:t}}=T(this,Ke).CreateUniformBuffer("Text",{label:T(_u,cr)+" Uniform Buffer"}),{buffer:n,Font:{color:r,back:i,subpx:o,hint:a}}=T(this,Ke).CreateUniformBuffer("Font",{label:T(_u,cr)+" Font Uniform Buffer"});j(this,yo,t),j(this,Ji,e),j(this,Ts,s),j(this,Es,n),o[0]=+T(this,Il),a[0]=+T(this,kl),j(this,go,r),j(this,mo,i)},s0=function(){T(this,Tn).forEach(s=>s.destroy()),T(this,Tn).splice(0),j(this,hr,new Float32Array(30*T(this,ri).length)),T(this,Tn).push(T(this,Ke).CreateVertexBuffer(T(this,hr)))},qd=function(){const{cap_height:s,ascent:e,x_height:t,descent:n,line_gap:r}=T(this,si),i=T(this,rr)/s;return{capScale:i,ascent:Math.round(e*i),lowScale:Math.round(t*i)/t,lineHeight:Math.round((e+n+r)*i+T(this,ii))}},Hd=function([s,e],t,n,r=0){const{aspect:i,ix:o,descent:a,iy:l,row_height:u}=T(this,si),{flags:c,bearing_x:h,rect:d,advance_x:w}=n,{lowScale:k,capScale:A,ascent:m}=t,S=1&c?k:A,b=i*S,f=s+b*(h+r-o),v=f+b*(d[2]-d[0]),_=e-m-S*(a+l),E=_+S*u,D=S*l*2;return{position:[s+=w*b,e],vertices:[f,E,d[0],d[1],D,v,E,d[2],d[1],D,f,_,d[0],d[3],D,f,_,d[0],d[3],D,v,E,d[2],d[1],D,v,_,d[2],d[3],D]}},El=function(){const[s,e]=T(this,Ke).CanvasSize,t=Math.round(-.5*T(this,Fe).size[0]),n=Math.round(.5*T(this,Fe).size[1]),r=2/(2*Math.round(.5*s)),i=2/(2*Math.round(.5*e));vl.set(r,0,0,0,i,0,t*r,n*i,1,T(this,Ji)),T(this,Ke).WriteBuffer(T(this,Ts),T(this,Ji).buffer)},mc=function(s,e){let t=0,n=" ",r=e,i=0,o=0;const a=se(this,kn,qd).call(this),{lineHeight:l,capScale:u}=a,{space_advance:c,chars:h,kern:d}=T(this,si);for(;!(i===T(this,ri).length||s.length<=o);){let w=T(this,ri)[i++];if(w===`
`){t=Math.max(t,r[0]),r[1]-=l,r[0]=e[0],n=" ";continue}if(w===" "){r[0]+=c*u,n=" ";continue}let k=h[w];k||(k=h[w="?"]);const A=se(this,kn,Hd).call(this,r,a,k,d[n+w]);for(let m=0,S=A.vertices.length;m<S;++m)s[o++]=A.vertices[m];r=A.position,n=w}T(this,Fe).min[0]=e[0],T(this,Fe).min[1]=e[1],T(this,Fe).max[0]=t||r[0],T(this,Fe).max[1]=r[1]+l,T(this,Fe).size[0]=T(this,Fe).max[0]-T(this,Fe).min[0],T(this,Fe).size[1]=T(this,Fe).max[1]-T(this,Fe).min[1],T(this,Ke).WriteBuffer(T(this,Tn)[0],s),se(this,kn,El).call(this)},ee(_u,cr,"SDFText");console.info("%cUWAL v0.0.12","background:#005a9c;padding:3px;color:#fff;");class lC{unet;sampsCount=0;image;Renderer;seedBuffer;color3f;color4u;totSamps=500;Computation;canvas;denoiserBuffer;denoiserFreq=25;storageBufferSize;workgroupDimension;resizeTimeout;seed;draw=this.render.bind(this);quartSamps=this.totSamps/4;context;constructor(){En.OnLost=()=>{},gA("/rt_ldr.tza").then(e=>this.unet=e)}resize(e,t){clearTimeout(this.resizeTimeout),this.resizeTimeout=setTimeout(()=>{En.Destroy([this.color3f.buffer,this.color4u.buffer]),this.create(this.Renderer.Canvas,e,t),this.setOutputCanvas(this.canvas,e,t),this.sampsCount=0},500)}setOutputCanvas(e,t,n){this.canvas=e,this.context=e.getContext("2d"),this.canvas.width=t,this.canvas.height=n,this.image=new ImageData(new Uint8ClampedArray(t*n*4),t,n)}async create(e,t,n){const r=Uint32Array.BYTES_PER_ELEMENT*4;return this.storageBufferSize=t*n*r,await this.checkRequiredLimits(e),this.Renderer=new(await En.Renderer(e)),this.Renderer.SetCanvasSize(t,n,!1),await this.createComputePipeline(),await this.createRenderPipeline(),requestAnimationFrame(this.draw),[t,n]}async checkRequiredLimits(e){const t=this.storageBufferSize*Uint32Array.BYTES_PER_ELEMENT*4;En.RequiredLimits={maxStorageBufferBindingSize:t},En.SetRequiredFeatures("bgra8unorm-storage");try{this.Computation=new(await En.Computation()),this.workgroupDimension=this.Computation.GetMaxEvenWorkgroupDimension(2)}catch(n){this.create(e,832,624),console.warn(n),console.warn(["Will be used a fallback with the minimum `maxStorageBufferBindingSize`","value available in all WebGPU contexts (134217728 bytes [128 MB]),","which produces a 832 x 624 pixel image."].join(" "))}}async createComputePipeline(){const e=this.createSpheres(),[t,n]=this.Renderer.CanvasSize,r=new this.Computation.Pipeline;await this.Computation.AddPipeline(r,{module:r.CreateShaderModule(`
                const SPHERES = ${e.length}u;
                ${yA}
            `),constants:{DIMENSION_SIZE:this.workgroupDimension,SAMPLES:4/this.totSamps}});const{seed:i,buffer:o}=r.CreateUniformBuffer("seed");this.color3f=r.CreateStorageBuffer("color3f",this.storageBufferSize*.75),this.denoiserBuffer=r.CreateReadableBuffer(this.storageBufferSize),this.color4u=r.CreateStorageBuffer("color4u",{length:this.storageBufferSize,usage:GPUBufferUsage.COPY_SRC});const{spheres:a,buffer:l}=r.CreateStorageBuffer("spheres",e.length);for(let u=0,c=0;u<e.length;u++,c=u*12)a[u].p.set(e[u].p,c+0),a[u].rad.set(e[u].rad,c+3),a[u].e.set(e[u].e,c+4),a[u].refl.set(e[u].refl,c+7),a[u].c.set(e[u].c,c+8);this.Computation.WriteBuffer(l,a[0].p.buffer),this.seedBuffer=o,this.seed=i,r.SetBindGroups(r.CreateBindGroup(r.CreateBindGroupEntries([this.Renderer.ResolutionBuffer,this.seedBuffer,this.color3f.buffer,this.color4u.buffer,l]))),this.Computation.Workgroups=[t/this.workgroupDimension,n/this.workgroupDimension]}async createRenderPipeline(){const e=new this.Renderer.Pipeline;await this.Renderer.AddPipeline(e,e.CreateShaderModule([Od.Resolution,Od.Quad,bA])),e.SetBindGroups(e.CreateBindGroup(e.CreateBindGroupEntries([this.Renderer.ResolutionBuffer,this.color4u.buffer]))),e.SetDrawParams(6)}async render(){this.updateSeedAndSampsBuffer(),this.Computation.Compute(!1),this.Computation.CopyBufferToBuffer(this.color4u.buffer,this.denoiserBuffer),this.Computation.Submit(),this.Renderer.Render(),++this.sampsCount%this.denoiserFreq||await this.denoise(),this.sampsCount<this.quartSamps&&requestAnimationFrame(this.draw)}updateSeedAndSampsBuffer(){this.seed[0]=Math.random()*4294967295,this.seed[1]=Math.random()*4294967295,this.seed[2]=Math.random()*4294967295,this.Computation.WriteBuffer(this.seedBuffer,this.seed)}async denoise(){await this.denoiserBuffer.mapAsync(GPUMapMode.READ),this.image.data.set(new Uint32Array(this.denoiserBuffer.getMappedRange())),this.denoiserBuffer.unmap();const e=this.context;this.unet.tileExecute({done:t=>e.putImageData(t,0,0),color:this.image})}createSpheres(){return[{p:[998,40.8,81.6],rad:[1e3],e:[0,0,0],refl:[0],c:[.8,.2,.2]},{p:[-898,40.8,81.6],rad:[1e3],e:[0,0,0],refl:[0],c:[.2,.2,.8]},{p:[50,40.8,1e3],rad:[1e3],e:[0,0,0],refl:[0],c:[.2,.8,.2]},{p:[50,40.8,-830],rad:[1e3],e:[0,0,0],refl:[0],c:[0,0,0]},{p:[50,1e3,81.6],rad:[1e3],e:[0,0,0],refl:[0],c:[.8,.8,.8]},{p:[50,-1e3+81.6+4.2,81.6],rad:[1e3],e:[0,0,0],refl:[0],c:[.8,.8,.8]},{p:[27,16.5,47],rad:[16.5],e:[0,0,0],refl:[1],c:[.999,.999,.999]},{p:[73,16.5,78],rad:[16.5],e:[0,0,0],refl:[2],c:[.999,.999,.999]},{p:[50,68.16-.27+74.2,81.6],rad:[60],e:[12,12,12],refl:[0],c:[0,0,0]}]}}const vu=new lC;self.onmessage=async({data:s})=>{const{width:e,height:t}=s;switch(s.action){case"Transfer::WebGPU":const[n,r]=await vu.create(s.canvas,e,t);(e!==n||t!==r)&&self.postMessage({width:n,height:r});break;case"Transfer::2D":return vu.setOutputCanvas(s.canvas,e,t);case"Resize::Window":return vu.resize(e,t)}};self.onerror=console.error;
