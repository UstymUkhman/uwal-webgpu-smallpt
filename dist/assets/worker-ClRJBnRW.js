function hm(s,e){return e.forEach(function(t){t&&typeof t!="string"&&!Array.isArray(t)&&Object.keys(t).forEach(function(n){if(n!=="default"&&!(n in s)){var r=Object.getOwnPropertyDescriptor(t,n);Object.defineProperty(s,n,r.get?r:{enumerable:!0,get:function(){return t[n]}})}})}),Object.freeze(s)}class fm{dims=[];paddedDims=[];layout="x";dataType="Float32";getByteSize(){let e=1;for(const t of this.paddedDims)e*=t;return this.dataType==="Float32"?e*=4:this.dataType==="Float16"&&(e*=2),e}}class dm{desc;data;constructor(e,t){this.desc=e,this.data=t}}class pm{_view;offset=0;constructor(e){this._view=e}read(e){const t=this._view,n=this.offset;switch(this.offset+=e,e){case 1:return t.getUint8(n);case 2:return t.getUint16(n,!0);case 4:return t.getUint32(n,!0);case 8:return Number(t.getBigUint64(n,!0));default:throw new Error("unsupported read size")}}}function mm(s){const e=new Uint8Array(s),t=new pm(new DataView(s));if(t.read(2)!==16855)throw new Error("invalid or corrupted weights blob");const r=t.read(1);if(t.read(1),r!==2)throw new Error("unsupported weights blob version");const i=t.read(8);t.offset=i;const o=t.read(4),a=new Map;for(let l=0;l<o;++l){const u=new fm,c=t.read(2),h=new TextDecoder().decode(e.subarray(t.offset,t.offset+c));t.offset+=c;const d=t.read(1);for(let S=0;S<d;++S)u.dims.push(t.read(4));u.paddedDims=[...u.dims],new TextDecoder().decode(e.subarray(t.offset,t.offset+d))==="oihw"&&(u.layout="oihw"),t.offset+=d;const I=String.fromCharCode(t.read(1));if(I==="f")u.dataType="Float32";else if(I==="h")u.dataType="Float16";else throw new Error("invalid tensor data type");const E=t.read(8),m=e.slice(E,E+u.getByteSize());a.set(h,new dm(u,m))}return a}/**
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
 */const gm=1e-7,ym=1e-4;class bm{constructor(e,t){this.backend=e,this.dataMover=t,this.data=new WeakMap,this.dataIdsCount=0}get(e){return this.data.has(e)||this.dataMover.moveData(this.backend,e),this.data.get(e)}set(e,t){this.dataIdsCount++,this.data.set(e,t)}has(e){return this.data.has(e)}delete(e){return this.dataIdsCount--,this.data.delete(e)}numDataIds(){return this.dataIdsCount}}class lh{refCount(e){return xt("refCount")}incRef(e){return xt("incRef")}timerAvailable(){return!0}time(e){return xt("time")}read(e){return xt("read")}readSync(e){return xt("readSync")}readToGPU(e,t){return xt("readToGPU")}numDataIds(){return xt("numDataIds")}disposeData(e,t){return xt("disposeData")}write(e,t,n){return xt("write")}move(e,t,n,r,i){return xt("move")}createTensorFromGPUData(e,t,n){return xt("createTensorFromGPUData")}memory(){return xt("memory")}floatPrecision(){return xt("floatPrecision")}epsilon(){return this.floatPrecision()===32?gm:ym}dispose(){return xt("dispose")}}function xt(s){throw new Error(`'${s}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}/**
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
 */function wm(s){let e=s.length,t=0;for(;e>0;)t=Math.random()*e|0,e--,vs(s,e,t)}function vs(s,e,t){const n=s[e];s[e]=s[t],s[t]=n}function xm(s){let e=0;for(let t=0;t<s.length;t++)e+=s[t];return e}function P(s,e){if(!s)throw new Error(typeof e=="string"?e:e())}function vm(s,e,t=""){P(Ht(s,e),()=>t+` Shapes ${s} and ${e} must match`)}function uh(s){P(s!=null,()=>"The input to the tensor constructor must be a non-null value.")}function he(s){if(s.length===0)return 1;let e=s[0];for(let t=1;t<s.length;t++)e*=s[t];return e}function Ht(s,e){if(s===e)return!0;if(s==null||e==null||s.length!==e.length)return!1;for(let t=0;t<s.length;t++)if(s[t]!==e[t])return!1;return!0}function ra(s){return s%1===0}function ci(s,e){return e<=s.length?s:s+" ".repeat(e-s.length)}function _m(s,e){let t=1,n=-1;for(let i=0;i<s.length;++i)if(s[i]>=0)t*=s[i];else if(s[i]===-1){if(n!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${n} and dim ${i}`);n=i}else if(s[i]<0)throw Error(`Shapes can not be < 0. Found ${s[i]} at dim ${i}`);if(n===-1){if(e>0&&e!==t)throw Error(`Size(${e}) must match the product of shape ${s}`);return s}if(t===0)throw Error(`Cannot infer the missing size in [${s}] when there are 0 elements`);if(e%t!==0)throw Error(`The implicit shape can't be a fractional number. Got ${e} / ${t}`);const r=s.slice();return r[n]=e/t,r}function Ar(s,e){const t=e.length;return s=s==null?e.map((n,r)=>r):[].concat(s),P(s.every(n=>n>=-t&&n<t),()=>`All values in axis param must be in range [-${t}, ${t}) but got axis ${s}`),P(s.every(n=>ra(n)),()=>`All values in axis param must be integers but got axis ${s}`),s.map(n=>n<0?t+n:n)}function Sm(s,e){const t=[],n=[],r=e!=null&&Array.isArray(e)&&e.length===0,i=e==null||r?null:Ar(e,s).sort();let o=0;for(let a=0;a<s.length;++a){if(i!=null){if(i[o]===a&&s[a]!==1)throw new Error(`Can't squeeze axis ${a} since its dim '${s[a]}' is not 1`);(i[o]==null||i[o]>a)&&s[a]===1&&(t.push(s[a]),n.push(a)),i[o]<=a&&o++}s[a]!==1&&(t.push(s[a]),n.push(a))}return{newShape:t,keptDims:n}}function Cs(s,e){return He(s,e)}function He(s,e){let t=null;if(s==null||s==="float32")t=new Float32Array(e);else if(s==="int32")t=new Int32Array(e);else if(s==="bool")t=new Uint8Array(e);else if(s==="string")t=new Array(e);else throw new Error(`Unknown data type ${s}`);return t}function Im(s,e){for(let t=0;t<s.length;t++){const n=s[t];if(isNaN(n)||!isFinite(n))throw Error(`A tensor of type ${e} being uploaded contains ${n}.`)}}function km(s){return s==="bool"||s==="complex64"||s==="float32"||s==="int32"||s==="string"}function ia(s){if(s==="float32"||s==="int32")return 4;if(s==="complex64")return 8;if(s==="bool")return 1;throw new Error(`Unknown dtype ${s}`)}function Tm(s){if(s==null)return 0;let e=0;return s.forEach(t=>e+=t.length),e}function uo(s){return typeof s=="string"||s instanceof String}function Em(s){return typeof s=="boolean"}function oa(s){return typeof s=="number"}function Cr(s){return Array.isArray(s)?Cr(s[0]):s instanceof Float32Array?"float32":s instanceof Int32Array||s instanceof Uint8Array||s instanceof Uint8ClampedArray?"int32":oa(s)?"float32":uo(s)?"string":Em(s)?"bool":"float32"}function aa(s){return!!(s&&s.constructor&&s.call&&s.apply)}function Pt(s){const e=s.length;if(e<2)return[];const t=new Array(e-1);t[e-2]=s[e-1];for(let n=e-3;n>=0;--n)t[n]=t[n+1]*s[n+1];return t}function ch(s,e,t,n=!1){const r=new Array;if(e.length===1){const i=e[0]*(n?2:1);for(let o=0;o<i;o++)r[o]=t[s+o]}else{const i=e[0],o=e.slice(1),a=o.reduce((l,u)=>l*u)*(n?2:1);for(let l=0;l<i;l++)r[l]=ch(s+l*a,o,t,n)}return r}function su(s,e,t=!1){if(s.length===0)return e[0];const n=s.reduce((r,i)=>r*i)*(t?2:1);if(n===0)return[];if(n!==e.length)throw new Error(`[${s}] does not match the input size ${e.length}${t?" for a complex tensor":""}.`);return ch(0,s,e,t)}function Do(s,e){if(Array.isArray(s))return s;if(e==="float32")return s instanceof Float32Array?s:new Float32Array(s);if(e==="int32")return s instanceof Int32Array?s:new Int32Array(s);if(e==="bool"||e==="string")return Uint8Array.from(new Int32Array(s));throw new Error(`Unknown dtype ${e}`)}function hh(s,e){const t=Gn(s,e);for(let n=0;n<t.length;n++)t[n]=1;return t}function Gn(s,e){if(e==null||e==="float32"||e==="complex64")return new Float32Array(s);if(e==="int32")return new Int32Array(s);if(e==="bool")return new Uint8Array(s);throw new Error(`Unknown data type ${e}`)}function Nn(s){s.forEach(e=>{P(Number.isInteger(e)&&e>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${s}].`)})}function la(s,e,t){if(e===0)return 0;if(e===1)return s[0];let n=s[s.length-1];for(let r=0;r<s.length-1;++r)n+=t[r]*s[r];return n}function Ua(s,e,t){if(e===0)return[];if(e===1)return[s];const n=new Array(e);for(let r=0;r<n.length-1;++r)n[r]=Math.floor(s/t[r]),s-=n[r]*t[r];return n[n.length-1]=s,n}function za(s){return s&&s.then&&typeof s.then=="function"}/**
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
 */const ru="tfjsflags";class Am{constructor(e){this.global=e,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=Cm,this.populateURLFlags()}setPlatform(e,t){this.platform!=null&&(fe().getBool("IS_TEST")||fe().getBool("PROD")||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${e}.`)),this.platformName=e,this.platform=t}registerFlag(e,t,n){if(this.flagRegistry[e]={evaluationFn:t,setHook:n},this.urlFlags[e]!=null){const r=this.urlFlags[e];fe().getBool("IS_TEST")||fe().getBool("PROD")||console.warn(`Setting feature override from URL ${e}: ${r}.`),this.set(e,r)}}async getAsync(e){return e in this.flags?this.flags[e]:(this.flags[e]=await this.evaluateFlag(e),this.flags[e])}get(e){if(e in this.flags)return this.flags[e];const t=this.evaluateFlag(e);if(za(t))throw new Error(`Flag ${e} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[e]=t,this.flags[e]}getNumber(e){return this.get(e)}getBool(e){return this.get(e)}getString(e){return this.get(e)}getFlags(){return this.flags}get features(){return this.flags}set(e,t){if(this.flagRegistry[e]==null)throw new Error(`Cannot set flag ${e} as it has not been registered.`);this.flags[e]=t,this.flagRegistry[e].setHook!=null&&this.flagRegistry[e].setHook(t)}evaluateFlag(e){if(this.flagRegistry[e]==null)throw new Error(`Cannot evaluate flag '${e}': no evaluation function found.`);return this.flagRegistry[e].evaluationFn()}setFlags(e){this.flags=Object.assign({},e)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(typeof this.global>"u"||typeof this.global.location>"u"||typeof this.global.location.search>"u")return;const e=this.getQueryParams(this.global.location.search);ru in e&&e[ru].split(",").forEach(n=>{const[r,i]=n.split(":");this.urlFlags[r]=Nm(r,i)})}}function Cm(s){const e={};return s.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(t,...n)=>($m(e,n[0],n[1]),n.join("="))),e}function $m(s,e,t){s[decodeURIComponent(e)]=decodeURIComponent(t||"")}function Nm(s,e){const t=e.toLowerCase();return t==="true"||t==="false"?t==="true":`${+t}`===t?+t:e}function fe(){return fh}let fh=null;function Dm(s){fh=s}/**
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
 */let Oo;function dh(){if(Oo==null){let s;if(typeof window<"u")s=window;else if(typeof global<"u")s=global;else if(typeof process<"u")s=process;else if(typeof self<"u")s=self;else throw new Error("Could not find a global object");Oo=s}return Oo}function Om(){const s=dh();return s._tfGlobals==null&&(s._tfGlobals=new Map),s._tfGlobals}function Va(s,e){const t=Om();if(t.has(s))return t.get(s);{const n=e();return t.set(s,n),t.get(s)}}const Mm="Abs",ph="Add",Pm="All",Rm="ArgMax",Lm="AvgPool",Bm="AvgPool3D",Fm="BatchMatMul",Um="Bincount",mh="Cast",zm="ClipByValue",Vm="Complex",Gm="ComplexAbs",gh="Concat",Wm="Conv2D",qm="Conv2DBackpropFilter",Hm="Conv2DBackpropInput",jm="Conv3D",Km="Conv3DBackpropInputV2",Xm="CropAndResize",Ym="DepthwiseConv2dNative",Qm="RealDiv",Zm="Einsum",Jm="Elu",eg="Erf",tg="Equal",ng="Exp",sg="ExpandDims",rg="Fill",ig="FlipLeftRight",og="Floor",ag="FloorDiv",lg="GatherV2",ug="Greater",cg="GreaterEqual",Ga="Identity",hg="Imag",fg="LeakyRelu",dg="Less",pg="LessEqual",mg="Log",gg="Log1p",yg="LogicalAnd",bg="Max",wg="Maximum",yh="MaxPool",xg="MaxPool3D",vg="Mean",_g="Min",Sg="Minimum",Ig="MirrorPad",kg="Multiply",Tg="Neg",Eg="NonMaxSuppressionV3",Ag="NonMaxSuppressionV4",Cg="NonMaxSuppressionV5",$g="OnesLike",Ng="OneHot",Dg="Pack",bh="PadV2",Og="Pow",Mg="Prelu",Pg="Range",Rg="Real",Lg="Relu",Bg="Reshape",wh="ResizeNearestNeighbor",Fg="ResizeBilinear",Ug="Relu6",zg="Round",Vg="Select",Gg="Selu",xh="Slice",Wg="Sigmoid",qg="Softplus",Hg="Sqrt",jg="Sum",Kg="SplitV",Xg="Softmax",Yg="Sub",Qg="Tanh",vh="Tile",Zg="Transform",Mo="Transpose",Jg="Unpack",ey="ZerosLike",ty="Step",ny="RotateWithOffset",ua="FusedConv2D";/**
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
 */function _s(...s){fe().getBool("IS_TEST")||fe().getBool("PROD")||console.warn(...s)}/**
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
 */const Ti=Va("kernelRegistry",()=>new Map),sy=Va("gradRegistry",()=>new Map);function iu(s,e){const t=_h(s,e);return Ti.get(t)}function ou(s){return sy.get(s)}function au(s){const e=Ti.entries(),t=[];for(;;){const{done:n,value:r}=e.next();if(n)break;const[i,o]=r,[a]=i.split("_");a===s&&t.push(o)}return t}function ry(s){const{kernelName:e,backendName:t}=s,n=_h(e,t);Ti.has(n)&&_s(`The kernel '${e}' for backend '${t}' is already registered`),Ti.set(n,s)}function _h(s,e){return`${e}_${s}`}/**
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
 */function Sh(s){return s instanceof Float32Array||s instanceof Int32Array||s instanceof Uint8Array||s instanceof Uint8ClampedArray}function iy(s){return s&&s.__esModule&&Object.prototype.hasOwnProperty.call(s,"default")?s.default:s}function oy(s){if(Object.prototype.hasOwnProperty.call(s,"__esModule"))return s;var e=s.default;if(typeof e=="function"){var t=function n(){var r=!1;try{r=this instanceof n}catch{}return r?Reflect.construct(e,arguments,this.constructor):e.apply(this,arguments)};t.prototype=e.prototype}else t={};return Object.defineProperty(t,"__esModule",{value:!0}),Object.keys(s).forEach(function(n){var r=Object.getOwnPropertyDescriptor(s,n);Object.defineProperty(t,n,r.get?r:{enumerable:!0,get:function(){return s[n]}})}),t}var Po,lu;function ay(){if(lu)return Po;lu=1,Po=e;var s=null;try{s=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch{}function e(A,g,p){this.low=A|0,this.high=g|0,this.unsigned=!!p}e.prototype.__isLong__,Object.defineProperty(e.prototype,"__isLong__",{value:!0});function t(A){return(A&&A.__isLong__)===!0}e.isLong=t;var n={},r={};function i(A,g){var p,y,x;return g?(A>>>=0,(x=0<=A&&A<256)&&(y=r[A],y)?y:(p=a(A,(A|0)<0?-1:0,!0),x&&(r[A]=p),p)):(A|=0,(x=-128<=A&&A<128)&&(y=n[A],y)?y:(p=a(A,A<0?-1:0,!1),x&&(n[A]=p),p))}e.fromInt=i;function o(A,g){if(isNaN(A))return g?b:S;if(g){if(A<0)return b;if(A>=I)return N}else{if(A<=-E)return O;if(A+1>=E)return T}return A<0?o(-A,g).neg():a(A%w|0,A/w|0,g)}e.fromNumber=o;function a(A,g,p){return new e(A,g,p)}e.fromBits=a;var l=Math.pow;function u(A,g,p){if(A.length===0)throw Error("empty string");if(A==="NaN"||A==="Infinity"||A==="+Infinity"||A==="-Infinity")return S;if(typeof g=="number"?(p=g,g=!1):g=!!g,p=p||10,p<2||36<p)throw RangeError("radix");var y;if((y=A.indexOf("-"))>0)throw Error("interior hyphen");if(y===0)return u(A.substring(1),g,p).neg();for(var x=o(l(p,8)),k=S,C=0;C<A.length;C+=8){var R=Math.min(8,A.length-C),z=parseInt(A.substring(C,C+R),p);if(R<8){var j=o(l(p,R));k=k.mul(j).add(o(z))}else k=k.mul(x),k=k.add(o(z))}return k.unsigned=g,k}e.fromString=u;function c(A,g){return typeof A=="number"?o(A,g):typeof A=="string"?u(A,g):a(A.low,A.high,typeof g=="boolean"?g:A.unsigned)}e.fromValue=c;var h=65536,d=1<<24,w=h*h,I=w*w,E=I/2,m=i(d),S=i(0);e.ZERO=S;var b=i(0,!0);e.UZERO=b;var f=i(1);e.ONE=f;var _=i(1,!0);e.UONE=_;var v=i(-1);e.NEG_ONE=v;var T=a(-1,2147483647,!1);e.MAX_VALUE=T;var N=a(-1,-1,!0);e.MAX_UNSIGNED_VALUE=N;var O=a(0,-2147483648,!1);e.MIN_VALUE=O;var $=e.prototype;return $.toInt=function(){return this.unsigned?this.low>>>0:this.low},$.toNumber=function(){return this.unsigned?(this.high>>>0)*w+(this.low>>>0):this.high*w+(this.low>>>0)},$.toString=function(g){if(g=g||10,g<2||36<g)throw RangeError("radix");if(this.isZero())return"0";if(this.isNegative())if(this.eq(O)){var p=o(g),y=this.div(p),x=y.mul(p).sub(this);return y.toString(g)+x.toInt().toString(g)}else return"-"+this.neg().toString(g);for(var k=o(l(g,6),this.unsigned),C=this,R="";;){var z=C.div(k),j=C.sub(z.mul(k)).toInt()>>>0,G=j.toString(g);if(C=z,C.isZero())return G+R;for(;G.length<6;)G="0"+G;R=""+G+R}},$.getHighBits=function(){return this.high},$.getHighBitsUnsigned=function(){return this.high>>>0},$.getLowBits=function(){return this.low},$.getLowBitsUnsigned=function(){return this.low>>>0},$.getNumBitsAbs=function(){if(this.isNegative())return this.eq(O)?64:this.neg().getNumBitsAbs();for(var g=this.high!=0?this.high:this.low,p=31;p>0&&(g&1<<p)==0;p--);return this.high!=0?p+33:p+1},$.isZero=function(){return this.high===0&&this.low===0},$.eqz=$.isZero,$.isNegative=function(){return!this.unsigned&&this.high<0},$.isPositive=function(){return this.unsigned||this.high>=0},$.isOdd=function(){return(this.low&1)===1},$.isEven=function(){return(this.low&1)===0},$.equals=function(g){return t(g)||(g=c(g)),this.unsigned!==g.unsigned&&this.high>>>31===1&&g.high>>>31===1?!1:this.high===g.high&&this.low===g.low},$.eq=$.equals,$.notEquals=function(g){return!this.eq(g)},$.neq=$.notEquals,$.ne=$.notEquals,$.lessThan=function(g){return this.comp(g)<0},$.lt=$.lessThan,$.lessThanOrEqual=function(g){return this.comp(g)<=0},$.lte=$.lessThanOrEqual,$.le=$.lessThanOrEqual,$.greaterThan=function(g){return this.comp(g)>0},$.gt=$.greaterThan,$.greaterThanOrEqual=function(g){return this.comp(g)>=0},$.gte=$.greaterThanOrEqual,$.ge=$.greaterThanOrEqual,$.compare=function(g){if(t(g)||(g=c(g)),this.eq(g))return 0;var p=this.isNegative(),y=g.isNegative();return p&&!y?-1:!p&&y?1:this.unsigned?g.high>>>0>this.high>>>0||g.high===this.high&&g.low>>>0>this.low>>>0?-1:1:this.sub(g).isNegative()?-1:1},$.comp=$.compare,$.negate=function(){return!this.unsigned&&this.eq(O)?O:this.not().add(f)},$.neg=$.negate,$.add=function(g){t(g)||(g=c(g));var p=this.high>>>16,y=this.high&65535,x=this.low>>>16,k=this.low&65535,C=g.high>>>16,R=g.high&65535,z=g.low>>>16,j=g.low&65535,G=0,X=0,Z=0,ne=0;return ne+=k+j,Z+=ne>>>16,ne&=65535,Z+=x+z,X+=Z>>>16,Z&=65535,X+=y+R,G+=X>>>16,X&=65535,G+=p+C,G&=65535,a(Z<<16|ne,G<<16|X,this.unsigned)},$.subtract=function(g){return t(g)||(g=c(g)),this.add(g.neg())},$.sub=$.subtract,$.multiply=function(g){if(this.isZero())return S;if(t(g)||(g=c(g)),s){var p=s.mul(this.low,this.high,g.low,g.high);return a(p,s.get_high(),this.unsigned)}if(g.isZero())return S;if(this.eq(O))return g.isOdd()?O:S;if(g.eq(O))return this.isOdd()?O:S;if(this.isNegative())return g.isNegative()?this.neg().mul(g.neg()):this.neg().mul(g).neg();if(g.isNegative())return this.mul(g.neg()).neg();if(this.lt(m)&&g.lt(m))return o(this.toNumber()*g.toNumber(),this.unsigned);var y=this.high>>>16,x=this.high&65535,k=this.low>>>16,C=this.low&65535,R=g.high>>>16,z=g.high&65535,j=g.low>>>16,G=g.low&65535,X=0,Z=0,ne=0,oe=0;return oe+=C*G,ne+=oe>>>16,oe&=65535,ne+=k*G,Z+=ne>>>16,ne&=65535,ne+=C*j,Z+=ne>>>16,ne&=65535,Z+=x*G,X+=Z>>>16,Z&=65535,Z+=k*j,X+=Z>>>16,Z&=65535,Z+=C*z,X+=Z>>>16,Z&=65535,X+=y*G+x*j+k*z+C*R,X&=65535,a(ne<<16|oe,X<<16|Z,this.unsigned)},$.mul=$.multiply,$.divide=function(g){if(t(g)||(g=c(g)),g.isZero())throw Error("division by zero");if(s){if(!this.unsigned&&this.high===-2147483648&&g.low===-1&&g.high===-1)return this;var p=(this.unsigned?s.div_u:s.div_s)(this.low,this.high,g.low,g.high);return a(p,s.get_high(),this.unsigned)}if(this.isZero())return this.unsigned?b:S;var y,x,k;if(this.unsigned){if(g.unsigned||(g=g.toUnsigned()),g.gt(this))return b;if(g.gt(this.shru(1)))return _;k=b}else{if(this.eq(O)){if(g.eq(f)||g.eq(v))return O;if(g.eq(O))return f;var C=this.shr(1);return y=C.div(g).shl(1),y.eq(S)?g.isNegative()?f:v:(x=this.sub(g.mul(y)),k=y.add(x.div(g)),k)}else if(g.eq(O))return this.unsigned?b:S;if(this.isNegative())return g.isNegative()?this.neg().div(g.neg()):this.neg().div(g).neg();if(g.isNegative())return this.div(g.neg()).neg();k=S}for(x=this;x.gte(g);){y=Math.max(1,Math.floor(x.toNumber()/g.toNumber()));for(var R=Math.ceil(Math.log(y)/Math.LN2),z=R<=48?1:l(2,R-48),j=o(y),G=j.mul(g);G.isNegative()||G.gt(x);)y-=z,j=o(y,this.unsigned),G=j.mul(g);j.isZero()&&(j=f),k=k.add(j),x=x.sub(G)}return k},$.div=$.divide,$.modulo=function(g){if(t(g)||(g=c(g)),s){var p=(this.unsigned?s.rem_u:s.rem_s)(this.low,this.high,g.low,g.high);return a(p,s.get_high(),this.unsigned)}return this.sub(this.div(g).mul(g))},$.mod=$.modulo,$.rem=$.modulo,$.not=function(){return a(~this.low,~this.high,this.unsigned)},$.and=function(g){return t(g)||(g=c(g)),a(this.low&g.low,this.high&g.high,this.unsigned)},$.or=function(g){return t(g)||(g=c(g)),a(this.low|g.low,this.high|g.high,this.unsigned)},$.xor=function(g){return t(g)||(g=c(g)),a(this.low^g.low,this.high^g.high,this.unsigned)},$.shiftLeft=function(g){return t(g)&&(g=g.toInt()),(g&=63)===0?this:g<32?a(this.low<<g,this.high<<g|this.low>>>32-g,this.unsigned):a(0,this.low<<g-32,this.unsigned)},$.shl=$.shiftLeft,$.shiftRight=function(g){return t(g)&&(g=g.toInt()),(g&=63)===0?this:g<32?a(this.low>>>g|this.high<<32-g,this.high>>g,this.unsigned):a(this.high>>g-32,this.high>=0?0:-1,this.unsigned)},$.shr=$.shiftRight,$.shiftRightUnsigned=function(g){if(t(g)&&(g=g.toInt()),g&=63,g===0)return this;var p=this.high;if(g<32){var y=this.low;return a(y>>>g|p<<32-g,p>>>g,this.unsigned)}else return g===32?a(p,0,this.unsigned):a(p>>>g-32,0,this.unsigned)},$.shru=$.shiftRightUnsigned,$.shr_u=$.shiftRightUnsigned,$.toSigned=function(){return this.unsigned?a(this.low,this.high,!1):this},$.toUnsigned=function(){return this.unsigned?this:a(this.low,this.high,!0)},$.toBytes=function(g){return g?this.toBytesLE():this.toBytesBE()},$.toBytesLE=function(){var g=this.high,p=this.low;return[p&255,p>>>8&255,p>>>16&255,p>>>24,g&255,g>>>8&255,g>>>16&255,g>>>24]},$.toBytesBE=function(){var g=this.high,p=this.low;return[g>>>24,g>>>16&255,g>>>8&255,g&255,p>>>24,p>>>16&255,p>>>8&255,p&255]},e.fromBytes=function(g,p,y){return y?e.fromBytesLE(g,p):e.fromBytesBE(g,p)},e.fromBytesLE=function(g,p){return new e(g[0]|g[1]<<8|g[2]<<16|g[3]<<24,g[4]|g[5]<<8|g[6]<<16|g[7]<<24,p)},e.fromBytesBE=function(g,p){return new e(g[4]<<24|g[5]<<16|g[6]<<8|g[7],g[0]<<24|g[1]<<16|g[2]<<8|g[3],p)},Po}var Ih=ay(),kh=iy(Ih),ly=hm({__proto__:null,default:kh},[Ih]);/**
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
 */const Xn=kh||ly;function co(s){return Xn.fromString(s,!0,16)}const Th=co("c3a5c85c97cb3127"),Kn=co("b492b66fbe98f273"),ct=co("9ae16a3b2f90404f");function ca(s){return s.xor(s.shru(47))}function Eh(s,e,t){const n=s.slice(e,e+t);return Xn.fromBytes(Array.from(n),!0,!0)}function Te(s,e){return Eh(s,e,8)}function uu(s,e){return Eh(s,e,4)}function qe(s,e){return e===0?s:s.shru(e).or(s.shl(64-e))}function Bn(s,e,t=co("9ddfea08eb382d69")){let n=s.xor(e).mul(t);n=n.xor(n.shru(47));let r=e.xor(n).mul(t);return r=r.xor(r.shru(47)),r=r.mul(t),r}function uy(s,e,t,n,r,i){r=r.add(s),i=qe(i.add(r).add(n),21);const o=r;return r=r.add(e),r=r.add(t),i=i.add(qe(r,44)),[r.add(n),i.add(o)]}function Kr(s,e,t,n){return uy(Te(s,e),Te(s,e+8),Te(s,e+16),Te(s,e+24),t,n)}function cy(s,e=s.length){if(e>=8){const t=ct.add(e*2),n=Te(s,0).add(ct),r=Te(s,e-8),i=qe(r,37).mul(t).add(n),o=qe(n,25).add(r).mul(t);return Bn(i,o,t)}if(e>=4){const t=ct.add(e*2),n=uu(s,0);return Bn(n.shl(3).add(e),uu(s,e-4),t)}if(e>0){const t=s[0],n=s[e>>1],r=s[e-1],i=t+(n<<8),o=e+(r<<2);return ca(ct.mul(i).xor(Th.mul(o))).mul(ct)}return ct}function hy(s,e=s.length){const t=ct.add(e*2),n=Te(s,0).mul(Kn),r=Te(s,8),i=Te(s,e-8).mul(t),o=Te(s,e-16).mul(ct);return Bn(qe(n.add(r),43).add(qe(i,30)).add(o),n.add(qe(r.add(ct),18)).add(i),t)}function fy(s,e=s.length){const t=ct.add(e*2),n=Te(s,0).mul(ct),r=Te(s,8),i=Te(s,e-8).mul(t),o=Te(s,e-16).mul(ct),a=qe(n.add(r),43).add(qe(i,30)).add(o),l=Bn(a,n.add(qe(r.add(ct),18)).add(i),t),u=Te(s,16).mul(t),c=Te(s,24),h=a.add(Te(s,e-32)).mul(t),d=l.add(Te(s,e-24)).mul(t);return Bn(qe(u.add(c),43).add(qe(h,30)).add(d),u.add(qe(c.add(n),18)).add(h),t)}function dy(s,e=s.length){const t=Xn.fromNumber(81,!0);if(e<=32)return e<=16?cy(s,e):hy(s,e);if(e<=64)return fy(s,e);let n=t,r=t.mul(Kn).add(113),i=ca(r.mul(ct).add(113)).mul(ct),o=[Xn.UZERO,Xn.UZERO],a=[Xn.UZERO,Xn.UZERO];n=n.mul(ct).add(Te(s,0));let l=0;const u=(e-1>>6)*64,c=u+(e-1&63)-63;do n=qe(n.add(r).add(o[0]).add(Te(s,l+8)),37).mul(Kn),r=qe(r.add(o[1]).add(Te(s,l+48)),42).mul(Kn),n=n.xor(a[1]),r=r.add(o[0]).add(Te(s,l+40)),i=qe(i.add(a[0]),33).mul(Kn),o=Kr(s,l,o[1].mul(Kn),n.add(a[0])),a=Kr(s,l+32,i.add(a[1]),r.add(Te(s,l+16))),[i,n]=[n,i],l+=64;while(l!==u);const h=Kn.add(i.and(255).shl(1));return l=c,a[0]=a[0].add(e-1&63),o[0]=o[0].add(a[0]),a[0]=a[0].add(o[0]),n=qe(n.add(r).add(o[0]).add(Te(s,l+8)),37).mul(h),r=qe(r.add(o[1]).add(Te(s,l+48)),42).mul(h),n=n.xor(a[1].mul(9)),r=r.add(o[0].mul(9).add(Te(s,l+40))),i=qe(i.add(a[0]),33).mul(h),o=Kr(s,l,o[1].mul(h),n.add(a[0])),a=Kr(s,l+32,i.add(a[1]),r.add(Te(s,l+16))),[i,n]=[n,i],Bn(Bn(o[0],a[0],h).add(ca(r).mul(Th)).add(i),Bn(o[1],a[1],h).add(n),h)}/**
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
 */function py(s,e){return e==="string"?Zn(s):ho([s],e)}function my(s,e){return s instanceof Float32Array&&e==="float32"||s instanceof Int32Array&&e==="int32"||s instanceof Uint8Array&&e==="bool"}function ho(s,e){if(e==="string")throw new Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(s)&&(s=as(s)),fe().getBool("DEBUG")&&Im(s,e),my(s,e))return s;if(e==null||e==="float32"||e==="complex64")return new Float32Array(s);if(e==="int32")return new Int32Array(s);if(e==="bool"){const t=new Uint8Array(s.length);for(let n=0;n<t.length;++n)Math.round(s[n])!==0&&(t[n]=1);return t}else throw new Error(`Unknown data type ${e}`)}function $s(){return fe().platform.now()}function Zn(s,e="utf-8"){return e=e||"utf-8",fe().platform.encode(s,e)}function Ei(s,e="utf-8"){return e=e||"utf-8",fe().platform.decode(s,e)}function Wt(s){return fe().platform.isTypedArray!=null?fe().platform.isTypedArray(s):Sh(s)}function as(s,e=[],t=!1){if(e==null&&(e=[]),typeof s=="boolean"||typeof s=="number"||typeof s=="string"||za(s)||s==null||Wt(s)&&t)e.push(s);else if(Array.isArray(s)||Wt(s))for(let n=0;n<s.length;++n)as(s[n],e,t);else{let n=-1;for(const r of Object.keys(s))/^([1-9]+[0-9]*|0)$/.test(r)&&(n=Math.max(n,Number(r)));for(let r=0;r<=n;r++)as(s[r],e,t)}return e}/**
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
 */class gy{constructor(e,t){this.backendTimer=e,this.logger=t,t==null&&(this.logger=new by)}profileKernel(e,t,n){let r;const i=()=>{r=n()};let o;const a=$s();if(this.backendTimer.timerAvailable())o=this.backendTimer.time(i);else{i();for(const u of r)u.dataSync();o=Promise.resolve({kernelMs:$s()-a})}if(fe().getBool("CHECK_COMPUTATION_FOR_ERRORS"))for(let u=0;u<r.length;u++){const c=r[u];c.data().then(h=>{yy(h,c.dtype,e)})}return{kernelName:e,outputs:r,inputs:t,timeMs:o.then(u=>u.kernelMs),extraInfo:o.then(u=>u.getExtraProfileInfo!=null?u.getExtraProfileInfo():"")}}logKernelProfile(e){const{kernelName:t,outputs:n,timeMs:r,inputs:i,extraInfo:o}=e;n.forEach(a=>{Promise.all([a.data(),r,o]).then(l=>{this.logger.logKernelProfile(t,a,l[0],l[1],i,l[2])})})}}function yy(s,e,t){if(e!=="float32")return!1;for(let n=0;n<s.length;n++){const r=s[n];if(isNaN(r)||!isFinite(r))return console.warn(`Found ${r} in the result of '${t}'`),!0}return!1}class by{logKernelProfile(e,t,n,r,i,o){const a=typeof r=="number"?ci(`${r}ms`,9):r.error,l=ci(e,25),u=t.rank,c=t.size,h=ci(t.shape.toString(),14);let d="";for(const w in i){const I=i[w];if(I!=null){const E=I.shape||t.shape,m=E.length;d+=`${w}: ${m}D ${m>0?E:""} `}}console.log(`%c${l}	%c${a}	%c${u}D ${h}	%c${c}	%c${d}	%c${o}`,"font-weight:bold","color:red","color:blue","color: orange","color: green","color: steelblue")}}/**
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
 */function wy(s,e,t){const n={},r={};for(let l=0;l<e.length;l++)n[e[l].id]=!0;for(let l=0;l<s.length;l++){const u=s[l],c=u.inputs;for(const h in c){const d=c[h];let w=!1;for(let I=0;I<e.length;I++)if(n[d.id]){u.outputs.forEach(E=>n[E.id]=!0),w=!0,r[u.id]=!0;break}if(w)break}}const i={};i[t.id]=!0;const o={};for(let l=s.length-1;l>=0;l--){const u=s[l],c=u.inputs;for(let h=0;h<u.outputs.length;h++)if(i[u.outputs[h].id]){for(const d in c)i[c[d].id]=!0,o[u.id]=!0;break}}const a=[];for(let l=0;l<s.length;l++){const u=s[l];if(r[u.id]&&o[u.id]){const c={};for(const d in u.inputs){const w=u.inputs[d];n[w.id]&&(c[d]=w)}const h=Object.assign({},u);h.inputs=c,h.outputs=u.outputs,a.push(h)}}return a}function xy(s,e,t,n){for(let r=e.length-1;r>=0;r--){const i=e[r],o=[];if(i.outputs.forEach(l=>{const u=s[l.id];u!=null?o.push(u):o.push(null)}),i.gradient==null)throw new Error(`Cannot compute gradient: gradient function not found for ${i.kernelName}.`);const a=i.gradient(o);for(const l in i.inputs){if(!(l in a))throw new Error(`Cannot backprop through input ${l}. Available gradients found: ${Object.keys(a)}.`);const u=t(()=>a[l]());if(u.dtype!=="float32")throw new Error(`Error in gradient for op ${i.kernelName}. The gradient of input ${l} must have 'float32' dtype, but has '${u.dtype}'`);const c=i.inputs[l];if(!Ht(u.shape,c.shape))throw new Error(`Error in gradient for op ${i.kernelName}. The gradient of input '${l}' has shape '${u.shape}', which does not match the shape of the input '${c.shape}'`);if(s[c.id]==null)s[c.id]=u;else{const h=s[c.id];s[c.id]=n(h,u),h.dispose()}}}}/**
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
 */const cu=20,Qs=3,Ro=7;function vy(s,e,t,n){const r=Pt(e),i=_y(s,e,t,r),o=e.length,a=hi(s,e,t,r,i),l=["Tensor"];return n&&(l.push(`  dtype: ${t}`),l.push(`  rank: ${o}`),l.push(`  shape: [${e}]`),l.push("  values:")),l.push(a.map(u=>"    "+u).join(`
`)),l.join(`
`)}function _y(s,e,t,n){const r=he(e),i=n[n.length-1],o=new Array(i).fill(0),a=e.length,l=t==="complex64"?sr(s):s;if(a>1)for(let u=0;u<r/i;u++){const c=u*i;for(let h=0;h<i;h++)o[h]=Math.max(o[h],nr(l[c+h],0,t).length)}return o}function nr(s,e,t){let n;return Array.isArray(s)?n=`${parseFloat(s[0].toFixed(Ro))} + ${parseFloat(s[1].toFixed(Ro))}j`:uo(s)?n=`'${s}'`:t==="bool"?n=Ah(s):n=parseFloat(s.toFixed(Ro)).toString(),ci(n,e)}function Ah(s){return s===0?"false":"true"}function hi(s,e,t,n,r,i=!0){const o=t==="complex64"?2:1,a=e[0],l=e.length;if(l===0){if(t==="complex64"){const E=sr(s);return[nr(E[0],0,t)]}return t==="bool"?[Ah(s[0])]:[s[0].toString()]}if(l===1){if(a>cu){const m=Qs*o;let S=Array.from(s.slice(0,m)),b=Array.from(s.slice((a-Qs)*o,a*o));return t==="complex64"&&(S=sr(S),b=sr(b)),["["+S.map((f,_)=>nr(f,r[_],t)).join(", ")+", ..., "+b.map((f,_)=>nr(f,r[a-Qs+_],t)).join(", ")+"]"]}return["["+(t==="complex64"?sr(s):Array.from(s)).map((m,S)=>nr(m,r[S],t)).join(", ")+"]"]}const u=e.slice(1),c=n.slice(1),h=n[0]*o,d=[];if(a>cu){for(let E=0;E<Qs;E++){const m=E*h,S=m+h;d.push(...hi(s.slice(m,S),u,t,c,r,!1))}d.push("...");for(let E=a-Qs;E<a;E++){const m=E*h,S=m+h;d.push(...hi(s.slice(m,S),u,t,c,r,E===a-1))}}else for(let E=0;E<a;E++){const m=E*h,S=m+h;d.push(...hi(s.slice(m,S),u,t,c,r,E===a-1))}const w=l===2?",":"";d[0]="["+(a>0?d[0]+w:"");for(let E=1;E<d.length-1;E++)d[E]=" "+d[E]+w;let I=`,
`;for(let E=2;E<l;E++)I+=`
`;return d[d.length-1]=" "+d[d.length-1]+"]"+(i?"":I),d}function sr(s){const e=[];for(let t=0;t<s.length;t+=2)e.push([s[t],s[t+1]]);return e}/**
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
 */class Ai{constructor(e,t,n){if(this.dtype=t,this.shape=e.slice(),this.size=he(e),n!=null){const r=n.length;P(r===this.size,()=>`Length of values '${r}' does not match the size inferred by the shape '${this.size}'.`)}if(t==="complex64")throw new Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=n||He(t,this.size),this.strides=Pt(e)}set(e,...t){t.length===0&&(t=[0]),P(t.length===this.rank,()=>`The number of provided coordinates (${t.length}) must match the rank (${this.rank})`);const n=this.locToIndex(t);this.values[n]=e}get(...e){e.length===0&&(e=[0]);let t=0;for(const r of e){if(r<0||r>=this.shape[t]){const i=`Requested out of range element at ${e}.   Buffer shape=${this.shape}`;throw new Error(i)}t++}let n=e[e.length-1];for(let r=0;r<e.length-1;++r)n+=this.strides[r]*e[r];return this.values[n]}locToIndex(e){if(this.rank===0)return 0;if(this.rank===1)return e[0];let t=e[e.length-1];for(let n=0;n<e.length-1;++n)t+=this.strides[n]*e[n];return t}indexToLoc(e){if(this.rank===0)return[];if(this.rank===1)return[e];const t=new Array(this.shape.length);for(let n=0;n<t.length-1;++n)t[n]=Math.floor(e/this.strides[n]),e-=t[n]*this.strides[n];return t[t.length-1]=e,t}get rank(){return this.shape.length}toTensor(){return tn().makeTensor(this.values,this.shape,this.dtype)}}let tn=null,Ss=null;function Sy(s){tn=s}function Iy(s){Ss=s}class et{constructor(e,t,n,r){this.kept=!1,this.isDisposedInternal=!1,this.shape=e.slice(),this.dtype=t||"float32",this.size=he(e),this.strides=Pt(e),this.dataId=n,this.id=r,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){const e=await this.data();return Ss.buffer(this.shape,this.dtype,e)}bufferSync(){return Ss.buffer(this.shape,this.dtype,this.dataSync())}async array(){const e=await this.data();return su(this.shape,e,this.dtype==="complex64")}arraySync(){return su(this.shape,this.dataSync(),this.dtype==="complex64")}async data(){this.throwIfDisposed();const e=tn().read(this.dataId);if(this.dtype==="string"){const t=await e;try{return t.map(n=>Ei(n))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return e}dataToGPU(e){return this.throwIfDisposed(),tn().readToGPU(this.dataId,e)}dataSync(){this.throwIfDisposed();const e=tn().readSync(this.dataId);if(this.dtype==="string")try{return e.map(t=>Ei(t))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return e}async bytes(){this.throwIfDisposed();const e=await tn().read(this.dataId);return this.dtype==="string"?e:new Uint8Array(e.buffer)}dispose(){this.isDisposed||(this.kerasMask&&this.kerasMask.dispose(),tn().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw new Error("Tensor is disposed.")}print(e=!1){return Ss.print(this,e)}clone(){return this.throwIfDisposed(),Ss.clone(this)}toString(e=!1){const t=this.dataSync();return vy(t,this.shape,this.dtype,e)}cast(e){return this.throwIfDisposed(),Ss.cast(this,e)}variable(e=!0,t,n){return this.throwIfDisposed(),tn().makeVariable(this,e,t,n)}}Object.defineProperty(et,Symbol.hasInstance,{value:s=>!!s&&s.data!=null&&s.dataSync!=null&&s.throwIfDisposed!=null});function Ch(){return Va("Tensor",()=>et)}Ch();class Ci extends et{constructor(e,t,n,r){super(e.shape,e.dtype,e.dataId,r),this.trainable=t,this.name=n}assign(e){if(e.dtype!==this.dtype)throw new Error(`dtype of the new value (${e.dtype}) and previous value (${this.dtype}) must match`);if(!Ht(e.shape,this.shape))throw new Error(`shape of the new value (${e.shape}) and previous value (${this.shape}) must match`);tn().disposeTensor(this),this.dataId=e.dataId,tn().incRef(this,null)}dispose(){tn().disposeVariable(this),this.isDisposedInternal=!0}}Object.defineProperty(Ci,Symbol.hasInstance,{value:s=>s instanceof et&&s.assign!=null&&s.assign instanceof Function});/**
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
 */var hu;(function(s){s.R0="R0",s.R1="R1",s.R2="R2",s.R3="R3",s.R4="R4",s.R5="R5",s.R6="R6"})(hu||(hu={}));var ha;(function(s){s.float32="float32",s.int32="int32",s.bool="int32",s.complex64="complex64"})(ha||(ha={}));var fa;(function(s){s.float32="float32",s.int32="int32",s.bool="bool",s.complex64="complex64"})(fa||(fa={}));var da;(function(s){s.float32="float32",s.int32="float32",s.bool="float32",s.complex64="complex64"})(da||(da={}));var pa;(function(s){s.float32="complex64",s.int32="complex64",s.bool="complex64",s.complex64="complex64"})(pa||(pa={}));const ky={float32:da,int32:ha,bool:fa,complex64:pa};function Wa(s,e){if(s==="string"||e==="string"){if(s==="string"&&e==="string")return"string";throw new Error(`Can not upcast ${s} with ${e}`)}return ky[s][e]}function Ty(s){return Wa(s,"int32")}function $h(s){return s!=null&&typeof s=="object"&&"texture"in s&&s.texture instanceof WebGLTexture}function Nh(s){return typeof GPUBuffer<"u"&&s!=null&&typeof s=="object"&&"buffer"in s&&s.buffer instanceof GPUBuffer}/**
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
 */function bt(s,e){if(s.dtype===e.dtype)return[s,e];const t=Wa(s.dtype,e.dtype);return[s.cast(t),e.cast(t)]}function Dh(s){const e=[];return Oh(s,e,new Set),e}function Oh(s,e,t){if(s==null)return;if(s instanceof et){e.push(s);return}if(!Ey(s))return;const n=s;for(const r in n){const i=n[r];t.has(i)||(t.add(i),Oh(i,e,t))}}function Ey(s){return Array.isArray(s)||typeof s=="object"}/**
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
 */function Lo(s){return s.kernelName!=null}class fu{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(e=>e.name)))}}}dispose(){for(const e in this.registeredVariables)this.registeredVariables[e].dispose()}}class Ns{constructor(e){this.ENV=e,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new fu}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;const e=this.getSortedBackends();for(let t=0;t<e.length;t++){const n=e[t];if(await this.initializeBackend(n).success){await this.setBackend(n);return}}throw new Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(this.pendingBackendInit!=null)throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){const{name:e,asyncInit:t}=this.initializeBackendsAndReturnBest();if(t)throw new Error(`The highest priority backend '${e}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(e)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(e){if(!(e in this.registry))if(e in this.registryFactory){const{asyncInit:t}=this.initializeBackend(e);if(t)return null}else return null;return this.registry[e]}findBackendFactory(e){return e in this.registryFactory?this.registryFactory[e].factory:null}registerBackend(e,t,n=1){return e in this.registryFactory?(_s(`${e} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[e]={factory:t,priority:n},!0)}async setBackend(e){if(this.registryFactory[e]==null)throw new Error(`Backend name '${e}' not found in registry`);if(this.backendName=e,this.registry[e]==null){this.backendInstance=null;const{success:t,asyncInit:n}=this.initializeBackend(e);if(!(n?await t:t))return!1}return this.backendInstance=this.registry[e],this.setupRegisteredKernels(),this.profiler=new gy(this.backendInstance),!0}setupRegisteredKernels(){au(this.backendName).forEach(t=>{t.setupFunc!=null&&t.setupFunc(this.backendInstance)})}disposeRegisteredKernels(e){au(e).forEach(n=>{n.disposeFunc!=null&&n.disposeFunc(this.registry[e])})}initializeBackend(e){const t=this.registryFactory[e];if(t==null)throw new Error(`Cannot initialize backend ${e}, no registration found.`);try{const n=t.factory();if(n&&!(n instanceof lh)&&typeof n.then=="function"){const r=++this.pendingBackendInitId,i=n.then(o=>r<this.pendingBackendInitId?!1:(this.registry[e]=o,this.pendingBackendInit=null,!0)).catch(o=>(r<this.pendingBackendInitId||(this.pendingBackendInit=null,_s(`Initialization of backend ${e} failed`),_s(o.stack||o.message)),!1));return this.pendingBackendInit=i,{success:i,asyncInit:!0}}else return this.registry[e]=n,{success:!0,asyncInit:!1}}catch(n){return _s(`Initialization of backend ${e} failed`),_s(n.stack||n.message),{success:!1,asyncInit:!1}}}removeBackend(e){if(!(e in this.registryFactory))throw new Error(`${e} backend not found in registry`);this.backendName===e&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,e in this.registry&&(this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e]),delete this.registryFactory[e],this.backendName===e&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw new Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((e,t)=>this.registryFactory[t].priority-this.registryFactory[e].priority)}initializeBackendsAndReturnBest(){const e=this.getSortedBackends();for(let t=0;t<e.length;t++){const n=e[t],{success:r,asyncInit:i}=this.initializeBackend(n);if(i||r)return{name:n,asyncInit:i}}throw new Error("Could not initialize any backends, all backend initializations failed.")}moveData(e,t){const n=this.state.tensorInfo.get(t),r=n.backend,i=this.readSync(t),o=r.refCount(t);r.disposeData(t,!0),n.backend=e,e.move(t,i,n.shape,n.dtype,o),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(e,t){let n=null;if(t==null){if(typeof e!="function")throw new Error("Please provide a function to tidy()");t=e}else{if(typeof e!="string"&&!(e instanceof String))throw new Error("When calling with two arguments, the first argument to tidy() must be a string");if(typeof t!="function")throw new Error("When calling with two arguments, the 2nd argument to tidy() must be a function");n=e}let r;return this.scopedRun(()=>this.startScope(n),()=>this.endScope(r),()=>(r=t(),r instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),r))}scopedRun(e,t,n){e();try{const r=n();return t(),r}catch(r){throw t(),r}}nextTensorId(){return Ns.nextTensorId++}nextVariableId(){return Ns.nextVariableId++}clone(e){const t=H.runKernel(Ga,{x:e}),n={x:e},r=o=>({x:()=>{const a="float32",l={x:o},u={dtype:a};return H.runKernel(mh,l,u)}}),i=[];return this.addTapeNode(this.state.activeScope.name,n,[t],r,i,{}),t}runKernel(e,t,n){if(this.backendName==null&&this.backend,!(iu(e,this.backendName)!=null))throw new Error(`Kernel '${e}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:e,inputs:t,attrs:n})}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(e,t,n){const r=this.backend.numDataIds();let i=0;n.forEach(l=>{i+=l.dtype==="complex64"?3:1});const o=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],a=r-t-i-o;if(a>0)throw new Error(`Backend '${this.backendName}' has an internal memory leak (${a} data ids) after running '${e}'`)}runKernelFunc(e){let t,n=[];const r=this.isTapeOn(),i=this.state.numBytes,o=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let a;this.backendName==null&&this.backend;let l;const u=Lo(e)?e.kernelName:this.state.activeScope!=null?this.state.activeScope.name:"";if(Lo(e)){const{kernelName:I,inputs:E,attrs:m}=e;this.backendName==null&&this.backend;const S=iu(I,this.backendName);P(S!=null,()=>`Cannot find registered kernel '${I}' for backend '${this.backendName}'`),a=()=>{const b=this.backend.numDataIds();l=S.kernelFunc({inputs:E,attrs:m,backend:this.backend});const f=Array.isArray(l)?l:[l];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(I,b,f);const _=f.map(v=>v.rank!=null?v:this.makeTensorFromTensorInfo(v));if(r){const v=this.getTensorsForGradient(I,E,_);n=this.saveTensorsForBackwardMode(v)}return _}}else{const{forwardFunc:I}=e,E=m=>{r&&(n=m.map(S=>this.keep(this.clone(S))))};a=()=>{const m=this.backend.numDataIds();l=this.tidy(()=>I(this.backend,E));const S=Array.isArray(l)?l:[l];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(u,m,S),S}}const{inputs:c,attrs:h}=e,d=Lo(e)?null:e.backwardsFunc;let w;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool("DEBUG")&&!this.state.profiling?t=a():(w=this.profiler.profileKernel(u,c,()=>a()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(w),t=w.outputs)}),r&&this.addTapeNode(u,c,t,d,n,h),this.state.profiling&&this.state.activeProfile.kernels.push({name:u,bytesAdded:this.state.numBytes-i,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-o,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(c).map(I=>c[I]!=null?c[I].shape:null),outputShapes:t.map(I=>I.shape),kernelTimeMs:w.timeMs,extraInfo:w.extraInfo}),Array.isArray(l)?t:t[0]}saveTensorsForBackwardMode(e){return e.map(n=>this.keep(this.clone(n)))}getTensorsForGradient(e,t,n){const r=ou(e);if(r!=null){const i=r.inputsToSave||[],o=r.outputsToSave||[];let a;r.saveAllInputs?(P(Array.isArray(t),()=>"saveAllInputs is true, expected inputs to be an array."),a=Object.keys(t).map(u=>t[u])):a=i.map(u=>t[u]);const l=n.filter((u,c)=>o[c]);return a.concat(l)}return[]}makeTensor(e,t,n,r){if(e==null)throw new Error("Values passed to engine.makeTensor() are null");n=n||"float32",r=r||this.backend;let i=e;n==="string"&&uo(e[0])&&(i=e.map(l=>Zn(l)));const o=r.write(i,t,n),a=new et(t,n,o,this.nextTensorId());if(this.trackTensor(a,r),n==="string"){const l=this.state.tensorInfo.get(o),u=Tm(i);this.state.numBytes+=u-l.bytes,l.bytes=u}return a}makeTensorFromDataId(e,t,n,r){n=n||"float32";const i={dataId:e,shape:t,dtype:n};return this.makeTensorFromTensorInfo(i,r)}makeTensorFromTensorInfo(e,t){const{dataId:n,shape:r,dtype:i}=e,o=new et(r,i,n,this.nextTensorId());return this.trackTensor(o,t),o}makeVariable(e,t=!0,n,r){n=n||this.nextVariableId().toString(),r!=null&&r!==e.dtype&&(e=e.cast(r));const i=new Ci(e,t,n,this.nextTensorId());if(this.state.registeredVariables[i.name]!=null)throw new Error(`Variable with name ${i.name} was already registered`);return this.state.registeredVariables[i.name]=i,this.incRef(i,this.backend),i}trackTensor(e,t){this.state.numTensors++,e.dtype==="string"&&this.state.numStringTensors++;let n=0;e.dtype!=="complex64"&&e.dtype!=="string"&&(n=e.size*ia(e.dtype)),this.state.numBytes+=n,this.state.tensorInfo.has(e.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(e.dataId,{backend:t||this.backend,dtype:e.dtype,shape:e.shape,bytes:n})),e instanceof Ci||this.track(e)}incRef(e,t){this.trackTensor(e,t),this.backend.incRef(e.dataId)}removeDataId(e,t){this.state.tensorInfo.has(e)&&this.state.tensorInfo.get(e).backend===t&&(this.state.tensorInfo.delete(e),this.state.numDataBuffers--)}disposeTensor(e){if(!this.state.tensorInfo.has(e.dataId))return;const t=this.state.tensorInfo.get(e.dataId);if(this.state.numTensors--,e.dtype==="string"&&(this.state.numStringTensors--,this.state.numBytes-=t.bytes),e.dtype!=="complex64"&&e.dtype!=="string"){const n=e.size*ia(e.dtype);this.state.numBytes-=n}t.backend.disposeData(e.dataId)&&this.removeDataId(e.dataId,t.backend)}disposeVariables(){for(const e in this.state.registeredVariables){const t=this.state.registeredVariables[e];this.disposeVariable(t)}}disposeVariable(e){this.disposeTensor(e),this.state.registeredVariables[e.name]!=null&&delete this.state.registeredVariables[e.name]}memory(){const e=this.backend.memory();return e.numTensors=this.state.numTensors,e.numDataBuffers=this.state.numDataBuffers,e.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(e.unreliable=!0,e.reasons==null&&(e.reasons=[]),e.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),e}async profile(e){this.state.profiling=!0;const t=this.state.numBytes,n=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await e(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(r=>r.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-t,this.state.activeProfile.newTensors=this.state.numTensors-n;for(const r of this.state.activeProfile.kernels)r.kernelTimeMs=await r.kernelTimeMs,r.extraInfo=await r.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(e,t,n,r,i,o){const a={id:this.state.nextTapeNodeId++,kernelName:e,inputs:t,outputs:n,saved:i},l=ou(e);l!=null&&(r=l.gradFunc),r!=null&&(a.gradient=u=>(u=u.map((c,h)=>{if(c==null){const d=n[h],w=Gn(d.size,d.dtype);return this.makeTensor(w,d.shape,d.dtype)}return c}),r(u.length>1?u:u[0],i,o))),this.state.activeTape.push(a)}keep(e){return e.kept=!0,e}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(e){const t={track:[],name:"unnamed scope",id:this.state.nextScopeId++};e&&(t.name=e),this.state.scopeStack.push(t),this.state.activeScope=t}endScope(e){const t=Dh(e),n=new Set(t.map(i=>i.id));for(let i=0;i<this.state.activeScope.track.length;i++){const o=this.state.activeScope.track[i];!o.kept&&!n.has(o.id)&&o.dispose()}const r=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],t.forEach(i=>{!i.kept&&i.scopeId===r.id&&this.track(i)})}gradients(e,t,n,r=!1){if(P(t.length>0,()=>"gradients() received an empty list of xs."),n!=null&&n.dtype!=="float32")throw new Error(`dy must have 'float32' dtype, but has '${n.dtype}'`);const i=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",e));P(i instanceof et,()=>"The result y returned by f() must be a tensor.");const o=wy(this.state.activeTape,t,i);if(!r&&o.length===0&&t.length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{const a={};a[i.id]=n??Ay(i.shape),xy(a,o,u=>this.tidy(u),Cy);const l=t.map(u=>a[u.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(u=>{for(const c of u.saved)c.dispose()}),this.state.activeTape=null),{value:i,grads:l}})}customGrad(e){return P(aa(e),()=>"The f passed in customGrad(f) must be a function."),(...t)=>{P(t.every(a=>a instanceof et),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");let n;const r={};t.forEach((a,l)=>{r[l]=a});const i=(a,l)=>(n=e(...t,l),P(n.value instanceof et,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),P(aa(n.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),n.value),o=(a,l)=>{const u=n.gradFunc(a,l),c=Array.isArray(u)?u:[u];P(c.length===t.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),P(c.every(d=>d instanceof et),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");const h={};return c.forEach((d,w)=>{h[w]=()=>d}),h};return this.runKernelFunc({forwardFunc:i,backwardsFunc:o,inputs:r})}}readSync(e){return this.state.tensorInfo.get(e).backend.readSync(e)}read(e){return this.state.tensorInfo.get(e).backend.read(e)}readToGPU(e,t){return this.state.tensorInfo.get(e).backend.readToGPU(e,t)}async time(e){const t=$s(),n=await this.backend.time(e);return n.wallMs=$s()-t,n}track(e){return this.state.activeScope!=null&&(e.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(e)),e}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new fu;for(const e in this.registry)this.disposeRegisteredKernels(e),this.registry[e].dispose(),delete this.registry[e];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}Ns.nextTensorId=0;Ns.nextVariableId=0;function Ay(s){const e=hh(he(s),"float32");return H.makeTensor(e,s,"float32")}function Mh(){const s=dh();if(s._tfengine==null){const e=new Am(s);s._tfengine=new Ns(e)}return Dm(s._tfengine.ENV),Sy(()=>s._tfengine),s._tfengine}const H=Mh();function Cy(s,e){const t={a:s,b:e};return H.runKernel(ph,t)}/**
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
 */function fo(s,e){let t=s;if(Wt(s))return e==="string"?[]:[s.length];if($h(s)){const r=s.channels||"RGBA";return[s.height,s.width*r.length]}else if(Nh(s))return[s.buffer.size/(e==null?4:ia(e))];if(!Array.isArray(s))return[];const n=[];for(;Array.isArray(t)||Wt(t)&&e!=="string";)n.push(t.length),t=t[0];return Array.isArray(s)&&fe().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&Ph(s,n,[]),n}function Ph(s,e,t){if(t=t||[],!Array.isArray(s)&&!Wt(s)){P(e.length===0,()=>`Element arr[${t.join("][")}] is a primitive, but should be an array/TypedArray of ${e[0]} elements`);return}P(e.length>0,()=>`Element arr[${t.join("][")}] should be a primitive, but is an array of ${s.length} elements`),P(s.length===e[0],()=>`Element arr[${t.join("][")}] should have ${e[0]} elements, but has ${s.length} elements`);const n=e.slice(1);for(let r=0;r<s.length;++r)Ph(s[r],n,t.concat(r))}function du(s,e,t,n){if(s!=="string_or_numeric"){if(s==null)throw new Error("Expected dtype cannot be null.");if(s!=="numeric"&&s!==e||s==="numeric"&&e==="string")throw new Error(`Argument '${t}' passed to '${n}' must be ${s} tensor, but got ${e} tensor`)}}function q(s,e,t,n="numeric"){if(s instanceof Ch())return du(n,s.dtype,e,t),s;let r=Cr(s);if(r!=="string"&&["bool","int32","float32"].indexOf(n)>=0&&(r=n),du(n,r,e,t),s==null||!Wt(s)&&!Array.isArray(s)&&typeof s!="number"&&typeof s!="boolean"&&typeof s!="string"){const l=s==null?"null":s.constructor.name;throw new Error(`Argument '${e}' passed to '${t}' must be a Tensor or TensorLike, but got '${l}'`)}const i=fo(s,r);!Wt(s)&&!Array.isArray(s)&&(s=[s]);const a=r!=="string"?ho(s,r):as(s,[],!0);return H.makeTensor(a,i,r)}function Rh(s,e,t,n="numeric"){if(!Array.isArray(s))throw new Error(`Argument ${e} passed to ${t} must be a \`Tensor[]\` or \`TensorLike[]\``);return s.map((i,o)=>q(i,`${e}[${o}]`,t,n))}/**
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
 */function po(s,e,t,n){if(n==null)n=Cr(s);else if(n==="complex64")throw new Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if(Nh(s)||$h(s)){if(n!=="float32"&&n!=="int32")throw new Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${n}.`);return H.backend.createTensorFromGPUData(s,e||t,n)}if(!Wt(s)&&!Array.isArray(s)&&typeof s!="number"&&typeof s!="boolean"&&typeof s!="string")throw new Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(e!=null){Nn(e);const r=he(e),i=he(t);P(r===i,()=>`Based on the provided shape, [${e}], the tensor should have ${r} values but has ${i}`);for(let o=0;o<t.length;++o){const a=t[o],l=o===t.length-1?a!==he(e.slice(o)):!0;P(t[o]===e[o]||!l,()=>`Error creating a new Tensor. Inferred shape (${t}) does not match the provided shape (${e}). `)}}return!Wt(s)&&!Array.isArray(s)&&(s=[s]),e=e||t,s=n!=="string"?ho(s,n):as(s,[],!0),H.makeTensor(s,e,n)}/**
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
 */function fi(s,e,t){const n=fo(s,t);return po(s,e,n,t)}/**
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
 */function pt(s,e){uh(s);const t=fo(s,e);if(t.length!==1)throw new Error("tensor1d() requires values to be a flat/TypedArray");return po(s,null,t,e)}/**
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
 */const $y="__op";function Q(s){const e=Object.keys(s);if(e.length!==1)throw new Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${e.length} keys.`);let t=e[0];const n=s[t];t.endsWith("_")&&(t=t.substring(0,t.length-1)),t=t+$y;const r=(...i)=>{H.startScope(t);try{const o=n(...i);return za(o)&&console.error("Cannot return a Promise inside of tidy."),H.endScope(o),o}catch(o){throw H.endScope(null),o}};return Object.defineProperty(r,"name",{value:t,configurable:!0}),r}/**
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
 */function Ny(s,e,t=0){const n=q(s,"x","pad");if(n.rank===0)throw new Error("pad(scalar) is not defined. Pass non-scalar to pad");const r={paddings:e,constantValue:t},i={x:n};return H.runKernel(bh,i,r)}const Dy=Q({pad_:Ny});function Oy(s,e,t=0){return P(e.length===4&&e[0].length===2&&e[1].length===2&&e[2].length===2&&e[3].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),Dy(s,e,t)}const My=Q({pad4d_:Oy});/**
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
 */function Py(s,e,t){const n=q(s,"x","slice","string_or_numeric");if(n.rank===0)throw new Error("Slicing scalar is not possible");const r={x:n},i={begin:e,size:t};return H.runKernel(xh,r,i)}const Je=Q({slice_:Py});/**
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
 */function Ry(s,e,t){const n=q(s,"x","slice4d");return P(n.rank===4,()=>`slice4d expects a rank-4 tensor, but got a rank-${n.rank} tensor`),Je(n,e,t)}const yr=Q({slice4d_:Ry});/**
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
 */function Ly(s){const t={x:q(s,"x","clone","string_or_numeric")};return H.runKernel(Ga,t)}const Jn=Q({clone_:Ly});/**
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
 */function By(s,e=0){P(s.length>=1,()=>"Pass at least one tensor to concat");const t=Rh(s,"tensors","concat","string_or_numeric");if(t[0].dtype==="complex64"&&t.forEach(i=>{if(i.dtype!=="complex64")throw new Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${i.dtype}. `)}),t.length===1)return Jn(t[0]);const n=t,r={axis:e};return H.runKernel(gh,n,r)}const es=Q({concat_:By});function Fy(s,e){return es(s,e)}const Uy=Q({concat4d_:Fy});/**
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
 */function zy(){return typeof window<"u"&&window.document!=null||typeof WorkerGlobalScope<"u"}/**
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
 */const yt=fe();yt.registerFlag("DEBUG",()=>!1,s=>{s&&console.warn("Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.")});yt.registerFlag("IS_BROWSER",()=>zy());yt.registerFlag("IS_NODE",()=>typeof process<"u"&&typeof process.versions<"u"&&typeof process.versions.node<"u");yt.registerFlag("IS_CHROME",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor));yt.registerFlag("IS_SAFARI",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor));yt.registerFlag("PROD",()=>!1);yt.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>yt.getBool("DEBUG"));yt.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0);yt.registerFlag("IS_TEST",()=>!1);yt.registerFlag("CHECK_COMPUTATION_FOR_ERRORS",()=>yt.getBool("DEBUG"));yt.registerFlag("WRAP_TO_IMAGEBITMAP",()=>!1);yt.registerFlag("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU",()=>!1);yt.registerFlag("USE_SETTIMEOUTCUSTOM",()=>!1);/**
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
 */function Vy(s,e){const t=q(s,"real","complex"),n=q(e,"imag","complex");vm(t.shape,n.shape,`real and imag shapes, ${t.shape} and ${n.shape}, must match in call to tf.complex().`);const r={real:t,imag:n};return H.runKernel(Vm,r)}const qa=Q({complex_:Vy});class Bs{static join(e){return new Bs(e).slice()}constructor(e){if(this.shards=[],this.previousShardIndex=0,e==null||(e instanceof Array||(e=[e]),e=e.map(n=>Wt(n)?n.buffer:n),e.length===0))return;this.bufferUniformSize=e[0].byteLength;let t=0;for(let n=0;n<e.length;n++){const r=e[n];n!==e.length-1&&r.byteLength!==this.bufferUniformSize&&(this.bufferUniformSize=void 0);const i=t+r.byteLength;this.shards.push({buffer:r,start:t,end:i}),t=i}this.shards.length===0&&(this.byteLength=0),this.byteLength=this.shards[this.shards.length-1].end}slice(e=0,t=this.byteLength){if(this.shards.length===0)return new ArrayBuffer(0);if(e=isNaN(Number(e))?0:e,t=isNaN(Number(t))?0:t,e=Math.max(0,e),t=Math.min(this.byteLength,t),t<=e)return new ArrayBuffer(0);const n=this.findShardForByte(e);if(n===-1)throw new Error(`Could not find start shard for byte ${e}`);const r=t-e,i=new ArrayBuffer(r),o=new Uint8Array(i);let a=0;for(let l=n;l<this.shards.length;l++){const u=this.shards[l],h=e+a-u.start,d=a,I=Math.min(t,u.end)-u.start,E=new Uint8Array(u.buffer,h,I-h);if(o.set(E,d),a+=E.length,t<u.end)break}return i}findShardForByte(e){if(this.shards.length===0||e<0||e>=this.byteLength)return-1;if(this.bufferUniformSize!=null)return this.previousShardIndex=Math.floor(e/this.bufferUniformSize),this.previousShardIndex;function t(r){return e<r.start?-1:e>=r.end?1:0}if(t(this.shards[this.previousShardIndex])===0)return this.previousShardIndex;const n=Gy(this.shards,t);return n===-1?-1:(this.previousShardIndex=n,this.previousShardIndex)}}function Gy(s,e){let t=0,n=s.length;for(;t<=n;){const r=Math.floor((n-t)/2)+t,i=e(s[r]);if(i===0)return r;i<0?n=r:t=r+1}return-1}/**
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
 */function Bo(){return H}function pu(){return H.memory()}function Y(s,e){return H.tidy(s,e)}function Ce(s){Dh(s).forEach(t=>t.dispose())}function As(s){return H.keep(s)}function Wy(s,e,t=1){return H.registerBackend(s,e,t)}function qy(){return H.backend}/**
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
 */const mu=4;async function gu(s,e){const t=[],n=[],r=Array.isArray(s)?s.map(o=>o.name):Object.keys(s);for(let o=0;o<r.length;++o){const a=r[o],l=Array.isArray(s)?s[o].tensor:s[a];if(l.dtype!=="float32"&&l.dtype!=="int32"&&l.dtype!=="bool"&&l.dtype!=="string"&&l.dtype!=="complex64")throw new Error(`Unsupported dtype in weight '${a}': ${l.dtype}`);const u={name:a,shape:l.shape,dtype:l.dtype};if(l.dtype==="string"){const c=new Promise(async h=>{const d=await l.bytes(),w=d.reduce((m,S)=>m+S.length,0)+mu*d.length,I=new Uint8Array(w);let E=0;for(let m=0;m<d.length;m++){const S=d[m],b=new Uint8Array(new Uint32Array([S.length]).buffer);I.set(b,E),E+=mu,I.set(S,E),E+=S.length}h(I)});n.push(c)}else n.push(l.data());e!=null&&(u.group=e),t.push(u)}const i=await Promise.all(n);return{data:Hy(i),specs:t}}function Hy(s){if(s===null)throw new Error(`Invalid input value: ${JSON.stringify(s)}`);let e=0;const t=[];s.forEach(i=>{if(e+=i.byteLength,t.push(i.byteLength===i.buffer.byteLength?i:new i.constructor(i)),!(i instanceof Float32Array||i instanceof Int32Array||i instanceof Uint8Array))throw new Error(`Unsupported TypedArray subtype: ${i.constructor.name}`)});const n=new Uint8Array(e);let r=0;return t.forEach(i=>{n.set(new Uint8Array(i.buffer),r),r+=i.byteLength}),n.buffer}const Ha=typeof Buffer<"u"&&(typeof Blob>"u"||typeof atob>"u"||typeof btoa>"u");function yu(s){return Ha?Buffer.byteLength(s,"utf8"):new Blob([s]).size}function jy(s){if(Ha)return Buffer.from(s).toString("base64");const e=new Uint8Array(s);let t="";for(let n=0,r=e.length;n<r;n++)t+=String.fromCharCode(e[n]);return btoa(t)}function Ky(s){if(Ha){const n=Buffer.from(s,"base64");return n.buffer.slice(n.byteOffset,n.byteOffset+n.byteLength)}const e=atob(s),t=new Uint8Array(e.length);for(let n=0;n<e.length;++n)t.set([e.charCodeAt(n)],n);return t.buffer}function Xy(s){return Bs.join(s)}function Lh(s){if(s.modelTopology instanceof ArrayBuffer)throw new Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:s.modelTopology==null?0:yu(JSON.stringify(s.modelTopology)),weightSpecsBytes:s.weightSpecs==null?0:yu(JSON.stringify(s.weightSpecs)),weightDataBytes:s.weightData==null?0:new Bs(s.weightData).byteLength}}/**
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
 */class lt{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return lt.instance==null&&(lt.instance=new lt),lt.instance}static registerSaveRouter(e){lt.getInstance().saveRouters.push(e)}static registerLoadRouter(e){lt.getInstance().loadRouters.push(e)}static getSaveHandlers(e){return lt.getHandlers(e,"save")}static getLoadHandlers(e,t){return lt.getHandlers(e,"load",t)}static getHandlers(e,t,n){const r=[];return(t==="load"?lt.getInstance().loadRouters:lt.getInstance().saveRouters).forEach(o=>{const a=o(e,n);a!==null&&r.push(a)}),r}}const Yy=s=>lt.getSaveHandlers(s);/**
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
 */const ma="tensorflowjs",ga=1,Qn="models_store",Pn="model_info_store";function Bh(){if(!fe().getBool("IS_BROWSER"))throw new Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");const s=typeof window>"u"?self:window,e=s.indexedDB||s.mozIndexedDB||s.webkitIndexedDB||s.msIndexedDB||s.shimIndexedDB;if(e==null)throw new Error("The current browser does not appear to support IndexedDB.");return e}function ya(s){const e=s.result;e.createObjectStore(Qn,{keyPath:"modelPath"}),e.createObjectStore(Pn,{keyPath:"modelPath"})}class ls{constructor(e){if(this.indexedDB=Bh(),e==null||!e)throw new Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=e}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,e)}async load(){return this.databaseAction(this.modelPath)}databaseAction(e,t){return new Promise((n,r)=>{const i=this.indexedDB.open(ma,ga);i.onupgradeneeded=()=>ya(i),i.onsuccess=()=>{const o=i.result;if(t==null){const a=o.transaction(Qn,"readonly"),u=a.objectStore(Qn).get(this.modelPath);u.onsuccess=()=>{if(u.result==null)return o.close(),r(new Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));n(u.result.modelArtifacts)},u.onerror=c=>(o.close(),r(u.error)),a.oncomplete=()=>o.close()}else{t.weightData=Bs.join(t.weightData);const a=Lh(t),l=o.transaction(Pn,"readwrite");let u=l.objectStore(Pn),c;try{c=u.put({modelPath:this.modelPath,modelArtifactsInfo:a})}catch(d){return r(d)}let h;c.onsuccess=()=>{h=o.transaction(Qn,"readwrite");const d=h.objectStore(Qn);let w;try{w=d.put({modelPath:this.modelPath,modelArtifacts:t,modelArtifactsInfo:a})}catch(I){return r(I)}w.onsuccess=()=>n({modelArtifactsInfo:a}),w.onerror=I=>{u=l.objectStore(Pn);const E=u.delete(this.modelPath);E.onsuccess=()=>(o.close(),r(w.error)),E.onerror=m=>(o.close(),r(w.error))}},c.onerror=d=>(o.close(),r(c.error)),l.oncomplete=()=>{h==null?o.close():h.oncomplete=()=>o.close()}}},i.onerror=o=>r(i.error)})}}ls.URL_SCHEME="indexeddb://";const Fh=s=>fe().getBool("IS_BROWSER")&&!Array.isArray(s)&&s.startsWith(ls.URL_SCHEME)?Qy(s.slice(ls.URL_SCHEME.length)):null;lt.registerSaveRouter(Fh);lt.registerLoadRouter(Fh);function Qy(s){return new ls(s)}function Zy(s){return s.startsWith(ls.URL_SCHEME)?s.slice(ls.URL_SCHEME.length):s}class Jy{constructor(){this.indexedDB=Bh()}async listModels(){return new Promise((e,t)=>{const n=this.indexedDB.open(ma,ga);n.onupgradeneeded=()=>ya(n),n.onsuccess=()=>{const r=n.result,i=r.transaction(Pn,"readonly"),a=i.objectStore(Pn).getAll();a.onsuccess=()=>{const l={};for(const u of a.result)l[u.modelPath]=u.modelArtifactsInfo;e(l)},a.onerror=l=>(r.close(),t(a.error)),i.oncomplete=()=>r.close()},n.onerror=r=>t(n.error)})}async removeModel(e){return e=Zy(e),new Promise((t,n)=>{const r=this.indexedDB.open(ma,ga);r.onupgradeneeded=()=>ya(r),r.onsuccess=()=>{const i=r.result,o=i.transaction(Pn,"readwrite"),a=o.objectStore(Pn),l=a.get(e);let u;l.onsuccess=()=>{if(l.result==null)return i.close(),n(new Error(`Cannot find model with path '${e}' in IndexedDB.`));{const c=a.delete(e),h=()=>{u=i.transaction(Qn,"readwrite");const w=u.objectStore(Qn).delete(e);w.onsuccess=()=>t(l.result.modelArtifactsInfo),w.onerror=I=>n(l.error)};c.onsuccess=h,c.onerror=d=>(h(),i.close(),n(l.error))}},l.onerror=c=>(i.close(),n(l.error)),o.oncomplete=()=>{u==null?i.close():u.oncomplete=()=>i.close()}},r.onerror=i=>n(r.error)})}}/**
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
 */const En="/",Is="tensorflowjs_models",Uh="info",e0="model_topology",t0="weight_specs",n0="weight_data",s0="model_metadata";function zh(s){return{info:[Is,s,Uh].join(En),topology:[Is,s,e0].join(En),weightSpecs:[Is,s,t0].join(En),weightData:[Is,s,n0].join(En),modelMetadata:[Is,s,s0].join(En)}}function Vh(s){for(const e of Object.values(s))window.localStorage.removeItem(e)}function r0(s){const e=s.split(En);if(e.length<3)throw new Error(`Invalid key format: ${s}`);return e.slice(1,e.length-1).join(En)}function i0(s){return s.startsWith(us.URL_SCHEME)?s.slice(us.URL_SCHEME.length):s}class us{constructor(e){if(!fe().getBool("IS_BROWSER")||typeof window>"u"||typeof window.localStorage>"u")throw new Error("The current environment does not support local storage.");if(this.LS=window.localStorage,e==null||!e)throw new Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=e,this.keys=zh(this.modelPath)}async save(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{const t=JSON.stringify(e.modelTopology),n=JSON.stringify(e.weightSpecs),r=Lh(e),i=Bs.join(e.weightData);try{this.LS.setItem(this.keys.info,JSON.stringify(r)),this.LS.setItem(this.keys.topology,t),this.LS.setItem(this.keys.weightSpecs,n),this.LS.setItem(this.keys.weightData,jy(i));const o={format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,signature:e.signature!=null?e.signature:void 0,userDefinedMetadata:e.userDefinedMetadata!=null?e.userDefinedMetadata:void 0,modelInitializer:e.modelInitializer!=null?e.modelInitializer:void 0,initializerSignature:e.initializerSignature!=null?e.initializerSignature:void 0,trainingConfig:e.trainingConfig!=null?e.trainingConfig:void 0};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(o)),{modelArtifactsInfo:r}}catch{throw Vh(this.keys),new Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${r.modelTopologyBytes}, weightSpecsBytes=${r.weightSpecsBytes}, weightDataBytes=${r.weightDataBytes}.`)}}}async load(){const e=JSON.parse(this.LS.getItem(this.keys.info));if(e==null)throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);if(e.modelTopologyType!=="JSON")throw new Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");const t={},n=JSON.parse(this.LS.getItem(this.keys.topology));if(n==null)throw new Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);t.modelTopology=n;const r=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(r==null)throw new Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);t.weightSpecs=r;const i=this.LS.getItem(this.keys.modelMetadata);if(i!=null){const a=JSON.parse(i);t.format=a.format,t.generatedBy=a.generatedBy,t.convertedBy=a.convertedBy,a.signature!=null&&(t.signature=a.signature),a.userDefinedMetadata!=null&&(t.userDefinedMetadata=a.userDefinedMetadata),a.modelInitializer!=null&&(t.modelInitializer=a.modelInitializer),a.initializerSignature!=null&&(t.initializerSignature=a.initializerSignature),a.trainingConfig!=null&&(t.trainingConfig=a.trainingConfig)}const o=this.LS.getItem(this.keys.weightData);if(o==null)throw new Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return t.weightData=Ky(o),t}}us.URL_SCHEME="localstorage://";const Gh=s=>fe().getBool("IS_BROWSER")&&!Array.isArray(s)&&s.startsWith(us.URL_SCHEME)?o0(s.slice(us.URL_SCHEME.length)):null;lt.registerSaveRouter(Gh);lt.registerLoadRouter(Gh);function o0(s){return new us(s)}class a0{constructor(){P(fe().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),P(typeof window>"u"||typeof window.localStorage<"u",()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){const e={},t=Is+En,n=En+Uh;for(let r=0;r<this.LS.length;++r){const i=this.LS.key(r);if(i.startsWith(t)&&i.endsWith(n)){const o=r0(i);e[o]=JSON.parse(this.LS.getItem(i))}}return e}async removeModel(e){e=i0(e);const t=zh(e);if(this.LS.getItem(t.info)==null)throw new Error(`Cannot find model at path '${e}'`);const n=JSON.parse(this.LS.getItem(t.info));return Vh(t),n}}/**
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
 */const bu="://";class hn{constructor(){this.managers={}}static getInstance(){return hn.instance==null&&(hn.instance=new hn),hn.instance}static registerManager(e,t){P(e!=null,()=>"scheme must not be undefined or null."),e.endsWith(bu)&&(e=e.slice(0,e.indexOf(bu))),P(e.length>0,()=>"scheme must not be an empty string.");const n=hn.getInstance();P(n.managers[e]==null,()=>`A model store manager is already registered for scheme '${e}'.`),n.managers[e]=t}static getManager(e){const t=hn.getInstance().managers[e];if(t==null)throw new Error(`Cannot find model manager for scheme '${e}'`);return t}static getSchemes(){return Object.keys(hn.getInstance().managers)}}/**
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
 */class l0{constructor(){this.messageName="setTimeoutCustom",this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(e,t){return fetch(e,t)}now(){return performance.now()}encode(e,t){if(t!=="utf-8"&&t!=="utf8")throw new Error(`Browser's encoder only supports utf-8, but got ${t}`);return this.textEncoder==null&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(e)}decode(e,t){return new TextDecoder(t).decode(e)}setTimeoutCustom(e,t){if(typeof window>"u"||!fe().getBool("USE_SETTIMEOUTCUSTOM")){setTimeout(e,t);return}this.functionRefs.push(e),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},"*")},t),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener("message",n=>{if(n.source===window&&n.data.name===this.messageName){n.stopPropagation();const r=this.functionRefs[n.data.index];r(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}isTypedArray(e){return Sh(e)}}if(fe().get("IS_BROWSER")){fe().setPlatform("browser",new l0);try{hn.registerManager(us.URL_SCHEME,new a0)}catch{}try{hn.registerManager(ls.URL_SCHEME,new Jy)}catch{}}/**
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
 */const u0={importFetch:()=>require("node-fetch")};let Fo;class c0{constructor(){this.util=require("util"),this.textEncoder=new this.util.TextEncoder}fetch(e,t){return fe().global.fetch!=null?fe().global.fetch(e,t):(Fo==null&&(Fo=u0.importFetch()),Fo(e,t))}now(){const e=process.hrtime();return e[0]*1e3+e[1]/1e6}encode(e,t){if(t!=="utf-8"&&t!=="utf8")throw new Error(`Node built-in encoder only supports utf-8, but got ${t}`);return this.textEncoder.encode(e)}decode(e,t){return e.length===0?"":new this.util.TextDecoder(t).decode(e)}isTypedArray(e){return this.util.types.isFloat32Array(e)||this.util.types.isInt32Array(e)||this.util.types.isUint8Array(e)||this.util.types.isUint8ClampedArray(e)}}fe().get("IS_NODE")&&!fe().get("IS_BROWSER")&&fe().setPlatform("node",new c0);/**
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
 */function Ye(s,e="float32",t){return e=e||"float32",Nn(s),new Ai(s,e,t)}/**
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
 */function h0(s,e){const t=q(s,"x","cast");if(!km(e))throw new Error(`Failed to cast to unknown dtype ${e}`);if(e==="string"&&t.dtype!=="string"||e!=="string"&&t.dtype==="string")throw new Error("Only strings can be casted to strings");const n={x:t},r={dtype:e};return H.runKernel(mh,n,r)}const Ee=Q({cast_:h0});/**
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
 */function f0(s,e=!1){console.log(s.toString(e))}/**
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
 */Mh();const d0={buffer:Ye,cast:Ee,clone:Jn,print:f0};Iy(d0);/**
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
 */function p0(s,e){let t=q(s,"a","add"),n=q(e,"b","add");[t,n]=bt(t,n);const r={a:t,b:n};return H.runKernel(ph,r)}const ae=Q({add_:p0});/**
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
 */function m0(s,e){let t=q(s,"a","floorDiv"),n=q(e,"b","floorDiv");[t,n]=bt(t,n);const r={a:t,b:n};return H.runKernel(ag,r)}const g0=Q({floorDiv_:m0});/**
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
 */function y0(s,e){let t=q(s,"a","div"),n=q(e,"b","div");if([t,n]=bt(t,n),t.dtype==="int32"&&n.dtype==="int32")return g0(t,n);const r={a:t,b:n},i={};return H.runKernel(Qm,r,i)}const ge=Q({div_:y0});/**
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
 */function b0(s,e){let t=q(s,"a","mul"),n=q(e,"b","mul");[t,n]=bt(t,n);const r={a:t,b:n};return H.runKernel(kg,r)}const J=Q({mul_:b0});/**
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
 */function w0(s){const e=q(s,"x","abs");if(e.dtype==="complex64"){const t={x:e};return H.runKernel(Gm,t)}else{const t={x:e};return H.runKernel(Mm,t)}}const dt=Q({abs_:w0});/**
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
 */function x0(s,e=null,t=!1){const r={x:q(s,"x","all","bool")},i={axis:e,keepDims:t};return H.runKernel(Pm,r,i)}const v0=Q({all_:x0});/**
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
 */function _0(s,e=0){const n={x:q(s,"x","argMax")},r={axis:e};return H.runKernel(Rm,n,r)}const $i=Q({argMax_:_0});/**
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
 */function S0(s,e,t,n,r,i,o="channelsLast"){const[a,l]=br(e);let u;if(o==="channelsLast")u=[a,l,s[3],s[3]];else if(o==="channelsFirst")u=[a,l,s[1],s[1]];else throw new Error(`Unknown dataFormat ${o}`);return ja(s,u,t,n,r,i,!1,o)}function ja(s,e,t,n,r,i,o=!1,a="channelsLast"){let[l,u,c,h]=[-1,-1,-1,-1];if(a==="channelsLast")[l,u,c,h]=s;else if(a==="channelsFirst")[l,h,u,c]=s;else throw new Error(`Unknown dataFormat ${a}`);const[d,w,,I]=e,[E,m]=br(t),[S,b]=br(n),f=ba(d,S),_=ba(w,b),{padInfo:v,outHeight:T,outWidth:N}=T0(r,u,c,E,m,f,_,i,a),O=o?I*h:I;let $;return a==="channelsFirst"?$=[l,O,T,N]:a==="channelsLast"&&($=[l,T,N,O]),{batchSize:l,dataFormat:a,inHeight:u,inWidth:c,inChannels:h,outHeight:T,outWidth:N,outChannels:O,padInfo:v,strideHeight:E,strideWidth:m,filterHeight:d,filterWidth:w,effectiveFilterHeight:f,effectiveFilterWidth:_,dilationHeight:S,dilationWidth:b,inShape:s,outShape:$,filterShape:e}}function I0(s,e,t,n,r){n==null&&(n=k0(s,e,t));const i=s[0],o=s[1],a=Ni((i-e+2*n)/t+1,r),l=Ni((o-e+2*n)/t+1,r);return[a,l]}function k0(s,e,t,n=1){const r=ba(e,n);return Math.floor((s[0]*(t-1)-t+r)/2)}function br(s){return typeof s=="number"?[s,s,s]:s.length===2?[s[0],s[1],1]:s}function ba(s,e){return e<=1?s:s+(s-1)*(e-1)}function T0(s,e,t,n,r,i,o,a,l){let u,c,h;if(typeof s=="number"){u={top:s,bottom:s,left:s,right:s,type:s===0?"VALID":"NUMBER"};const w=I0([e,t],i,n,s,a);c=w[0],h=w[1]}else if(s==="same"){c=Math.ceil(e/n),h=Math.ceil(t/r);const d=Math.max(0,(c-1)*n+i-e),w=Math.max(0,(h-1)*r+o-t),I=Math.floor(d/2),E=d-I,m=Math.floor(w/2),S=w-m;u={top:I,bottom:E,left:m,right:S,type:"SAME"}}else if(s==="valid")u={top:0,bottom:0,left:0,right:0,type:"VALID"},c=Math.ceil((e-i+1)/n),h=Math.ceil((t-o+1)/r);else if(typeof s=="object"){const d=l==="channelsLast"?s[1][0]:s[2][0],w=l==="channelsLast"?s[1][1]:s[2][1],I=l==="channelsLast"?s[2][0]:s[3][0],E=l==="channelsLast"?s[2][1]:s[3][1];u={top:d,bottom:w,left:I,right:E,type:d===0&&w===0&&I===0&&E===0?"VALID":"EXPLICIT"},c=Ni((e-i+d+w)/n+1,a),h=Ni((t-o+I+E)/r+1,a)}else throw Error(`Unknown padding parameter: ${s}`);return{padInfo:u,outHeight:c,outWidth:h}}function Ni(s,e){if(!e)return Math.trunc(s);switch(e){case"round":return Math.round(s);case"ceil":return Math.ceil(s);case"floor":return Math.floor(s);default:throw new Error(`Unknown roundingMode ${e}`)}}function wa(s){const[e,t,n]=br(s);return e===1&&t===1&&n===1}function Fs(s,e){return wa(s)||wa(e)}function Ds(s){return br(s).every(e=>e>0)}function E0(s){if(s==="NHWC")return"channelsLast";if(s==="NCHW")return"channelsFirst";throw new Error(`Unknown dataFormat ${s}`)}function vn(s,e,t){if(t!=null){if(typeof e=="string")throw Error(`Error in ${s}: pad must be an integer when using dimRoundingMode ${t} but got pad ${e}.`);if(typeof e=="number")P(ra(e),()=>`Error in ${s}: pad must be an integer when using dimRoundingMode ${t} but got pad ${e}.`);else if(typeof e=="object")e.forEach(n=>{n.forEach(r=>{P(ra(r),()=>`Error in ${s}: pad must be an integer when using dimRoundingMode ${t} but got pad ${r}.`)})});else throw Error(`Error in ${s}: Unknown padding parameter: ${e}`)}}/**
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
 */function A0(s,e){const n={x:q(s,"x","reshape","string_or_numeric")},r={shape:e};return H.runKernel(Bg,n,r)}const se=Q({reshape_:A0});/**
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
 */function C0(s,e,t,n,r){const i=q(s,"x","avgPool","float32"),o=1;P(Fs(t,o),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${t} and dilations '${o}'`);let a=i,l=!1;i.rank===3&&(l=!0,a=se(i,[1,i.shape[0],i.shape[1],i.shape[2]])),P(a.rank===4,()=>`Error in avgPool: x must be rank 4 but got rank ${a.rank}.`),vn("avgPool",n,r);const u={x:a},c={filterSize:e,strides:t,pad:n,dimRoundingMode:r};let h=H.runKernel(Lm,u,c);return h=Ee(h,i.dtype),l?se(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const $0=Q({avgPool_:C0});/**
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
 */function N0(s,e,t,n,r,i="NDHWC"){const o=q(s,"x","avgPool3d","float32");let a=o,l=!1;o.rank===4&&(l=!0,a=se(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),P(a.rank===5,()=>`Error in avgPool3d: x must be rank 5 but got rank ${a.rank}.`),P(i==="NDHWC",()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${i}`),P(typeof t=="number"&&t>0||Array.isArray(t)&&t[0]>0&&t[1]>0&&t[2]>0,()=>`Error in avgPool3d: Stride must be > 0, but got '${t}'`),vn("avgPool3d",n,r);const u={x:a},c={filterSize:e,strides:t,pad:n,dimRoundingMode:r,dataFormat:i};let h=H.runKernel(Bm,u,c);return h=Ee(h,a.dtype),l?se(h,[h.shape[1],h.shape[2],h.shape[3],h.shape[4]]):h}const D0=Q({avgPool3d_:N0});/**
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
 */function O0(s,e,t=!1,n=!1){let r=q(s,"a","matMul"),i=q(e,"b","matMul");[r,i]=bt(r,i);const o={a:r,b:i},a={transposeA:t,transposeB:n};return H.runKernel(Fm,o,a)}const ln=Q({matMul_:O0});/**
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
 */function M0(s){const t={x:q(s,"x","sigmoid","float32")};return H.runKernel(Wg,t)}const Ka=Q({sigmoid_:M0});/**
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
 */function P0(s){const t={x:q(s,"x","tanh","float32")};return H.runKernel(Qg,t)}const Xa=Q({tanh_:P0});/**
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
 */function R0(s,e,t){const n=q(s,"x","bincount"),r=q(e,"weights","bincount");P(n.dtype==="int32",()=>`Error in bincount: input dtype must be int32, but got ${n.dtype}`),P(t>=0,()=>`size must be non-negative, but got ${t}.`),P(r.size===n.size||r.size===0,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${n.shape}, weights shape: ${r.shape}.`);const i={x:n,weights:r},o={size:t};return H.runKernel(Um,i,o)}const L0=Q({bincount_:R0});/**
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
 */function B0(s,e){let t=q(s,"broadcastTo","x");const n=t.shape;if(Nn(e),e.length<t.rank)throw new Error(`broadcastTo(): shape.length=${e.length} < input.rank=${t.rank}.`);if(e.length>t.rank){const u=t.shape.slice();for(;u.length<e.length;)u.unshift(1);t=se(t,u)}const r=t.shape,i=Array.from(e);for(let u=e.length-1;u>=0;u--)if(r[u]===e[u])i[u]=1;else if(t.shape[u]!==1)throw new Error(`broadcastTo(): [${n}] cannot be broadcast to [${e}].`);if(i.map((u,c)=>u>1?c:-1).filter(u=>u>=0).length===0)return Jn(t);const a={x:t},l={reps:i};return H.runKernel(vh,a,l)}const di=Q({broadcastTo_:B0});/**
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
 */function mo(s,e,t){Nn(s),t=t||Cr(e);const n={shape:s,value:e,dtype:t};return H.runKernel(rg,{},n)}/**
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
 */function F0(s,e,t){const n=q(s,"x","clipByValue");if(P(e<=t,()=>`Error in clip: min (${e}) must be less than or equal to max (${t}).`),e===t)return mo(n.shape,e,n.dtype);const r={x:n},i={clipValueMin:e,clipValueMax:t};return H.runKernel(zm,r,i)}const nn=Q({clipByValue_:F0});/**
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
 */function U0(s,e,t,n,r="NHWC",i=[1,1],o){const a=q(s,"x","conv2d","float32"),l=q(e,"filter","conv2d","float32");let u=a,c=!1;a.rank===3&&(c=!0,u=se(a,[1,a.shape[0],a.shape[1],a.shape[2]])),P(u.rank===4,()=>`Error in conv2d: input must be rank 4, but got rank ${u.rank}.`),P(l.rank===4,()=>`Error in conv2d: filter must be rank 4, but got rank ${l.rank}.`),vn("conv2d",n,o);const h=r==="NHWC"?u.shape[3]:u.shape[1];P(h===l.shape[2],()=>`Error in conv2d: depth of input (${h}) must match input depth for filter ${l.shape[2]}.`),P(Fs(t,i),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${t} and dilations '${i}'`),P(Ds(i),()=>"Error in conv2D: Dilated rates should be larger than 0."),P(Ds(t),()=>"Error in conv2D: Strides should be larger than 0.");const d={x:u,filter:l},w={strides:t,pad:n,dataFormat:r,dilations:i,dimRoundingMode:o},I=H.runKernel(Wm,d,w);return c?se(I,[I.shape[1],I.shape[2],I.shape[3]]):I}const Ya=Q({conv2d_:U0});function z0(s,e,t,n,r="NWC",i=1,o){const a=q(s,"x","conv1d"),l=q(e,"filter","conv1d");let u=a,c=!1;a.rank===2&&(c=!0,u=se(a,[1,a.shape[0],a.shape[1]])),P(u.rank===3,()=>`Error in conv1d: input must be rank 3, but got rank ${u.rank}.`),P(l.rank===3,()=>`Error in conv1d: filter must be rank 3, but got rank ${l.rank}.`),vn("conv1d",n,o),P(u.shape[2]===l.shape[1],()=>`Error in conv1d: depth of input (${u.shape[2]}) must match input depth for filter ${l.shape[1]}.`),P(Fs(t,i),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${t} and dilation '${i}'`),P(Ds(i),()=>"Error in conv1D: Dilated rates should be larger than 0."),P(Ds(t),()=>"Error in conv1D: Stride should be larger than 0."),P(r==="NWC",()=>`Error in conv1d: got dataFormat of ${r} but only NWC is currently supported.`);const h=se(l,[1,l.shape[0],l.shape[1],l.shape[2]]),d=se(u,[u.shape[0],1,u.shape[1],u.shape[2]]),m=Ya(d,h,[1,t],n,"NHWC",[1,i],o);return c?se(m,[m.shape[2],m.shape[3]]):se(m,[m.shape[0],m.shape[2],m.shape[3]])}const V0=Q({conv1d_:z0});/**
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
 */function G0(s,e,t,n,r,i="NHWC",o){P(s.length===e.rank,()=>`Length of inShape (${s.length}) and rank of dy (${e.rank}) must match`);let a=s,l=e,u=!1;e.rank===3&&(u=!0,l=se(e,[1,e.shape[0],e.shape[1],e.shape[2]]),a=[1,s[0],s[1],s[2]]),P(a.length===4,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${a.length}.`),P(l.rank===4,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${l.rank}`),P(t.rank===4,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${t.rank}`);const c=i==="NHWC"?a[3]:a[1],h=i==="NHWC"?l.shape[3]:l.shape[1];P(c===t.shape[2],()=>`Error in conv2dDerInput: depth of input (${c}) must match input depth for filter ${t.shape[2]}.`),P(h===t.shape[3],()=>`Error in conv2dDerInput: depth of output (${h}) must match output depth for filter ${t.shape[3]}.`),vn("conv2dDerInput",r,o);const d={dy:l,filter:t},w={strides:n,pad:r,dataFormat:i,dimRoundingMode:o,inputShape:a},I=H.runKernel(Hm,d,w);return u?se(I,[I.shape[1],I.shape[2],I.shape[3]]):I}const Wh=Q({conv2DBackpropInput_:G0});function W0(s,e,t,n,r,i){const o=q(s,"x","conv2dTranspose"),a=q(e,"filter","conv2dTranspose");return Wh(t,o,a,n,r,"NHWC",i)}const q0=Q({conv2dTranspose_:W0});/**
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
 */function H0(s,e,t,n,r="NDHWC",i=[1,1,1]){const o=q(s,"x","conv3d"),a=q(e,"filter","conv3d");let l=o,u=!1;o.rank===4&&(u=!0,l=se(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),P(l.rank===5,()=>`Error in conv3d: input must be rank 5, but got rank ${l.rank}.`),P(a.rank===5,()=>`Error in conv3d: filter must be rank 5, but got rank ${a.rank}.`),P(l.shape[4]===a.shape[3],()=>`Error in conv3d: depth of input (${l.shape[4]}) must match input depth for filter ${a.shape[3]}.`),P(Fs(t,i),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${t} and dilations '${i}'`),P(r==="NDHWC",()=>`Error in conv3d: got dataFormat of ${r} but only NDHWC is currently supported.`),P(Ds(i),()=>"Error in conv3D: Dilated rates should be larger than 0."),P(Ds(t),()=>"Error in conv3D: Strides should be larger than 0.");const c={x:l,filter:a},h={strides:t,pad:n,dataFormat:r,dilations:i},d=H.runKernel(jm,c,h);return u?se(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}const j0=Q({conv3d_:H0});/**
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
 */function K0(s,e,t,n,r){P(s.length===e.rank,()=>`Length of inShape (${s.length}) and rank of dy (${e.rank}) must match`);let i=s,o=e,a=!1;e.rank===4&&(a=!0,o=se(e,[1,e.shape[0],e.shape[1],e.shape[2],e.shape[3]]),i=[1,s[0],s[1],s[2],s[3]]);const l=i[4],u=o.shape[4];P(i.length===5,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${i.length}.`),P(o.rank===5,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${o.rank}`),P(t.rank===5,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${t.rank}`),P(l===t.shape[3],()=>`Error in conv3dDerInput: depth of input (${l}) must match input depth for filter ${t.shape[3]}.`),P(u===t.shape[4],()=>`Error in conv3dDerInput: depth of output (${u}) must match output depth for filter ${t.shape[4]}.`);const c={dy:o,filter:t},h={pad:r,strides:n,inputShape:i},d=H.runKernel(Km,c,h);return a?se(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}const X0=Q({conv3DBackpropInput_:K0});function Y0(s,e,t,n,r){const i=q(s,"x","conv3dTranspose"),o=q(e,"filter","conv3dTranspose");return X0(t,i,o,n,r)}const Q0=Q({conv3dTranspose_:Y0});/**
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
 */function Z0(s,e,t,n,r="NHWC",i=[1,1],o){const a=q(s,"x","depthwiseConv2d","float32"),l=q(e,"filter","depthwiseConv2d","float32");let u=a,c=!1;a.rank===3&&(c=!0,u=se(a,[1,a.shape[0],a.shape[1],a.shape[2]])),P(u.rank===4,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${u.rank}.`),P(l.rank===4,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${l.rank}.`);const h=r==="NHWC"?u.shape[3]:u.shape[1];P(h===l.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${h}) must match the inChannels dimension in filter ${l.shape[2]}.`),vn("depthwiseConv2d",n,o);const d={x:u,filter:l},w={strides:t,pad:n,dataFormat:r,dilations:i,dimRoundingMode:o},I=H.runKernel(Ym,d,w);return c?se(I,[I.shape[1],I.shape[2],I.shape[3]]):I}const J0=Q({depthwiseConv2d_:Z0});/**
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
 */function Di(s,e){const t=s.length,n=[];for(let r=0;r<t;r++){const i=t-1-r,o=s[i]||1;(e[e.length-1-r]||1)>1&&o===1&&n.unshift(i)}return n}function e1(s,e){const t=[];for(let n=0;n<e.length;n++){const r=s[s.length-n-1],i=e.length-n-1,o=e[i];(r==null||r===1&&o>1)&&t.unshift(i)}return t}function St(s,e){const t=Math.max(s.length,e.length),n=new Array(t);for(let r=0;r<t;r++){let i=s[s.length-r-1];i==null&&(i=1);let o=e[e.length-r-1];if(o==null&&(o=1),i===1)n[t-r-1]=o;else if(o===1)n[t-r-1]=i;else if(i!==o){const a=`Operands could not be broadcast together with shapes ${s} and ${e}.`;throw Error(a)}else n[t-r-1]=i}return n}/**
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
 */function t1(s,e){let t=q(s,"a","equal","string_or_numeric"),n=q(e,"b","equal","string_or_numeric");[t,n]=bt(t,n),St(t.shape,n.shape);const r={a:t,b:n};return H.runKernel(tg,r)}const cs=Q({equal_:t1});/**
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
 */function n1(s,e,t){const n=q(e,"a","where"),r=q(t,"b","where"),i=q(s,"condition","where","bool"),o=St(St(i.shape,n.shape),r.shape),a=di(i,o),l=di(n,o),u=di(r,o),c={condition:a,t:l,e:u};return H.runKernel(Vg,c)}const ts=Q({where_:n1});/**
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
 */function s1(s){const t={x:q(s,"x","zerosLike")};return H.runKernel(ey,t)}const bn=Q({zerosLike_:s1});/**
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
 */function r1(s,...e){const t=e.map((r,i)=>q(r,`tensors${i}`,"einsum")),n={equation:s};return H.runKernel(Zm,t,n)}const Zs=Q({einsum_:r1});/**
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
 */function i1(s){const t={x:q(s,"x","elu","float32")};return H.runKernel(Jm,t)}const qh=Q({elu_:i1});/**
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
 */function o1(s){let e=q(s,"x","erf");P(e.dtype==="int32"||e.dtype==="float32",()=>"Input dtype must be `int32` or `float32`."),e.dtype==="int32"&&(e=Ee(e,"float32"));const t={x:e};return H.runKernel(eg,t)}const a1=Q({erf_:o1});/**
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
 */function Hh(s,e){for(let t=0;t<s.length;++t)if(s[s.length-t-1]!==e-1-t)return!1;return!0}function l1(s,e,t){const n=s.length+e.length,r=[];let i=0,o=0;for(let a=0;a<n;a++)t.indexOf(a)===-1?r.push(s[i++]):r.push(e[o++]);return r}function Qa(s,e){const t=[],n=s.length;for(let i=0;i<n;i++)e.indexOf(i)===-1&&t.push(s[i]);const r=e.map(i=>s[i]);return[t,r]}function jh(s,e){const t=e.map(n=>1);return l1(s,t,e)}function u1(s,e,t){P(Hh(e,t),()=>`${s} supports only inner-most axes for now. Got axes ${e} and rank-${t} input.`)}function c1(s,e){if(Hh(s,e))return null;const t=[];for(let n=0;n<e;++n)s.indexOf(n)===-1&&t.push(n);return s.forEach(n=>t.push(n)),t}function h1(s,e){const t=[];for(let n=e-s;n<e;++n)t.push(n);return t}/**
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
 */function f1(s,e=null,t=!1){const r={x:q(s,"x","max")},i={reductionIndices:e,keepDims:t};return H.runKernel(bg,r,i)}const Fn=Q({max_:f1});/**
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
 */function d1(s,e=null,t=!1){const r={x:q(s,"x","min")},i={axis:e,keepDims:t};return H.runKernel(_g,r,i)}const wu=Q({min_:d1});/**
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
 */function p1(s,e){let t=q(s,"base","pow"),n=q(e,"exp","pow");[t,n]=bt(t,n);const r={a:t,b:n};return H.runKernel(Og,r)}const Oi=Q({pow_:p1});/**
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
 */function Rt(s,e){if((Wt(s)&&e!=="string"||Array.isArray(s))&&e!=="complex64")throw new Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if(e==="string"&&Wt(s)&&!(s instanceof Uint8Array))throw new Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return po(s,[],[],e)}/**
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
 */function m1(s){const t={x:q(s,"x","sqrt","float32")};return H.runKernel(Hg,t)}const sn=Q({sqrt_:m1});/**
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
 */function g1(s){const e=q(s,"x","square"),t={};return H.runKernel("Square",{x:e},t)}const Un=Q({square_:g1});/**
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
 */function y1(s,e=null,t=!1){let n=q(s,"x","sum");n.dtype==="bool"&&(n=Ee(n,"int32"));const r={x:n},i={axis:e,keepDims:t};return H.runKernel(jg,r,i)}const Se=Q({sum_:y1});/**
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
 */function b1(s,e="euclidean",t=null,n=!1){s=q(s,"x","norm");const r=Kh(s,e,t);let i=r.shape;if(n){const o=Ar(t,s.shape);i=jh(r.shape,o)}return se(r,i)}function Kh(s,e,t=null){if(s.rank===0)return dt(s);if(s.rank!==1&&t===null)return Kh(se(s,[-1]),e,t);if(s.rank===1||typeof t=="number"||Array.isArray(t)&&t.length===1){if(e===1)return Se(dt(s),t);if(e===1/0)return Fn(dt(s),t);if(e===-1/0)return wu(dt(s),t);if(e==="euclidean"||e===2)return sn(Se(Oi(dt(s),Rt(2,"int32")),t));throw new Error(`Error in norm: invalid ord value: ${e}`)}if(Array.isArray(t)&&t.length===2){if(e===1)return Fn(Se(dt(s),t[0]),t[1]-1);if(e===1/0)return Fn(Se(dt(s),t[1]),t[0]);if(e===-1/0)return wu(Se(dt(s),t[1]),t[0]);if(e==="fro"||e==="euclidean")return sn(Se(Un(s),t));throw new Error(`Error in norm: invalid ord value: ${e}`)}throw new Error(`Error in norm: invalid axis: ${t}`)}const Xh=Q({norm_:b1});/**
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
 */function w1(s){const t={x:q(s,"x","exp")};return H.runKernel(ng,t)}const xa=Q({exp_:w1});/**
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
 */function x1(s,e=0){const t=q(s,"x","expandDims","string_or_numeric");P(e<=t.rank,()=>"Axis must be <= rank of the tensor");const n={input:t},r={dim:e};return H.runKernel(sg,n,r)}const dn=Q({expandDims_:x1});/**
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
 */function v1(s,e){const t=q(s,"x","tile","string_or_numeric");P(t.rank===e.length,()=>`Error in transpose: rank of input ${t.rank} must match length of reps ${e}.`);const n={x:t},r={reps:e};return H.runKernel(vh,n,r)}const pi=Q({tile_:v1});/**
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
 */function _1(s,e,t,n="float32"){e==null&&(e=s);const r=Ye([s,e],n),i=s<=e?s:e;for(let a=0;a<i;++a)r.set(1,a,a);const o=se(r.toTensor(),[s,e]);if(t==null)return o;if(t.length===1)return pi(dn(o,0),[t[0],1,1]);if(t.length===2)return pi(dn(dn(o,0),0),[t[0],t[1],1,1]);if(t.length===3)return pi(dn(dn(dn(o,0),0),0),[t[0],t[1],t[2],1,1]);throw new Error(`eye() currently supports only 1D and 2D batchShapes, but received ${t.length}D.`)}const Yh=Q({eye_:_1});/**
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
 */function S1(s){const t={x:q(s,"x","floor","float32")};return H.runKernel(og,t)}const I1=Q({floor_:S1});/**
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
 */function k1(s,e,t=0,n=0){const r=q(s,"x","gather"),i=q(e,"indices","gather","int32"),o={x:r,indices:i},a={axis:t,batchDims:n};return H.runKernel(lg,o,a)}const T1=Q({gather_:k1});/**
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
 */function E1(s,e){let t=q(s,"a","greater","string_or_numeric"),n=q(e,"b","greater","string_or_numeric");[t,n]=bt(t,n),St(t.shape,n.shape);const r={a:t,b:n};return H.runKernel(ug,r)}const $r=Q({greater_:E1});/**
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
 */function A1(s,e){let t=q(s,"a","greaterEqual","string_or_numeric"),n=q(e,"b","greaterEqual","string_or_numeric");[t,n]=bt(t,n),St(t.shape,n.shape);const r={a:t,b:n};return H.runKernel(cg,r)}const C1=Q({greaterEqual_:A1});/**
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
 */function $1(s){const t={input:q(s,"input","imag")};return H.runKernel(hg,t)}const N1=Q({imag_:$1});/**
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
 */function D1(s,e=.2){const n={x:q(s,"x","leakyRelu")},r={alpha:e};return H.runKernel(fg,n,r)}const O1=Q({leakyRelu_:D1});/**
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
 */function M1(s,e){let t=q(s,"a","less","string_or_numeric"),n=q(e,"b","less","string_or_numeric");[t,n]=bt(t,n),St(t.shape,n.shape);const r={a:t,b:n};return H.runKernel(dg,r)}const xu=Q({less_:M1});/**
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
 */function P1(s,e){let t=q(s,"a","lessEqual","string_or_numeric"),n=q(e,"b","lessEqual","string_or_numeric");[t,n]=bt(t,n),St(t.shape,n.shape);const r={a:t,b:n};return H.runKernel(pg,r)}const Qh=Q({lessEqual_:P1});/**
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
 */function R1(s){const t={x:q(s,"x","log","float32")};return H.runKernel(mg,t)}const hs=Q({log_:R1});/**
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
 */function L1(s){const t={x:q(s,"x","log1p")};return H.runKernel(gg,t)}const B1=Q({log1p_:L1});/**
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
 */function F1(s,e){P(aa(s),()=>"The f passed in variableGrads(f) must be a function"),P(e==null||Array.isArray(e)&&e.every(u=>u instanceof Ci),()=>"The varList passed in variableGrads(f, varList) must be an array of variables");const t=e!=null;if(!t){e=[];for(const u in H.registeredVariables)e.push(H.registeredVariables[u])}const n=t?e.filter(u=>!u.trainable):null,r=e.length;e=e.filter(u=>u.trainable),P(e.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${r} variables is trainable.`);const i=!0,{value:o,grads:a}=H.gradients(s,e,null,i);P(a.some(u=>u!=null),()=>"Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize()."),P(o.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${o.rank} tensor`);const l={};return e.forEach((u,c)=>{a[c]!=null&&(l[u.name]=a[c])}),n?.forEach(u=>l[u.name]=null),{value:o,grads:l}}function va(s){return H.customGrad(s)}/**
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
 */function U1(s){const t={x:q(s,"x","neg")};return H.runKernel(Tg,t)}const Us=Q({neg_:U1});/**
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
 */function z1(s){const t={x:q(s,"x","softplus")};return H.runKernel(qg,t)}const Za=Q({softplus_:z1});/**
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
 */function V1(s,e){let t=q(s,"a","sub"),n=q(e,"b","sub");[t,n]=bt(t,n);const r={a:t,b:n};return H.runKernel(Yg,r)}const we=Q({sub_:V1});/**
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
 */function G1(s,e=-1){const t=q(s,"logits","logSoftmax");if(e===-1&&(e=t.rank-1),e!==t.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${t.rank} and axis was ${e}`);return va((r,i)=>{const a=Fn(r,e,!0),l=we(r,a),u=we(Ee(l,"float32"),hs(Se(xa(l),e,!0)));return i([u]),{value:u,gradFunc:(h,d)=>{const[w]=d,I=!0,E=xa(w);return we(h,J(Se(h,e,I),E))}}})(t)}const W1=Q({logSoftmax_:G1});/**
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
 */function q1(s,e){const t=q(s,"a","logicalAnd","bool"),n=q(e,"b","logicalAnd","bool");St(t.shape,n.shape);const r={a:t,b:n};return H.runKernel(yg,r)}const go=Q({logicalAnd_:q1});/**
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
 */function H1(s,e,t,n,r){const i=q(s,"x","maxPool"),o=1;let a=i,l=!1;i.rank===3&&(l=!0,a=se(i,[1,i.shape[0],i.shape[1],i.shape[2]])),P(a.rank===4,()=>`Error in maxPool: input must be rank 4 but got rank ${a.rank}.`),P(Fs(t,o),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${t} and dilations '${o}'`),vn("maxPool",n,r);const u={x:a},c={filterSize:e,strides:t,pad:n,dimRoundingMode:r},h=H.runKernel(yh,u,c);return l?se(h,[h.shape[1],h.shape[2],h.shape[3]]):h}const j1=Q({maxPool_:H1});/**
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
 */function K1(s,e=[1,1,1],t,n,r,i="NDHWC"){const o=q(s,"x","maxPool3d");let a=o,l=!1;o.rank===4&&(l=!0,a=se(o,[1,o.shape[0],o.shape[1],o.shape[2],o.shape[3]])),P(a.rank===5,()=>`Error in maxPool3d: x must be rank 5 but got rank ${a.rank}.`),P(i==="NDHWC",()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${i}`),vn("maxPool3d",n,r);const u={x:a},c={filterSize:e,strides:t,pad:n,dimRoundingMode:r,dataFormat:i},h=H.runKernel(xg,u,c);return l?se(h,[h.shape[1],h.shape[2],h.shape[3],h.shape[4]]):h}const X1=Q({maxPool3d_:K1});/**
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
 */function Y1(s,e){let t=q(s,"a","maximum"),n=q(e,"b","maximum");[t,n]=bt(t,n),t.dtype==="bool"&&(t=Ee(t,"int32"),n=Ee(n,"int32")),St(t.shape,n.shape);const r={a:t,b:n};return H.runKernel(wg,r)}const zs=Q({maximum_:Y1});/**
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
 */function Q1(s,e=null,t=!1){const r={x:q(s,"x","mean")},i={axis:e,keepDims:t};return H.runKernel(vg,r,i)}const Xe=Q({mean_:Q1});/**
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
 */function Os(s,e="float32"){if(Nn(s),e==="complex64"){const n=Os(s,"float32"),r=Os(s,"float32");return qa(n,r)}const t=Gn(he(s),e);return H.makeTensor(t,s,e)}/**
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
 */function Ja(s,e="float32"){if(Nn(s),e==="complex64"){const n=Ja(s,"float32"),r=Os(s,"float32");return qa(n,r)}const t=hh(he(s),e);return H.makeTensor(t,s,e)}/**
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
 */function Z1(s,e){let t=q(s,"a","minimum"),n=q(e,"b","minimum");[t,n]=bt(t,n),t.dtype==="bool"&&(t=Ee(t,"int32"),n=Ee(n,"int32")),St(t.shape,n.shape);const r={a:t,b:n};return H.runKernel(Sg,r)}const Mi=Q({minimum_:Z1});/**
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
 */function J1(s,e,t=1,n=0,r="int32"){if(e<2)throw new Error(`Error in oneHot: depth must be >=2, but it is ${e}`);const o={indices:q(s,"indices","oneHot","int32")},a={dtype:r,depth:e,onValue:t,offValue:n};return H.runKernel(Ng,o,a)}const eb=Q({oneHot_:J1});/**
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
 */function tb(s){const t={x:q(s,"x","onesLike")};return H.runKernel($g,t)}const Zh=Q({onesLike_:tb});/**
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
 */function nb(s,e){const t=q(s,"x","prelu"),n=q(e,"alpha","prelu"),r={x:t,alpha:n};return H.runKernel(Mg,r)}const sb=Q({prelu_:nb});var mi={exports:{}},rb=mi.exports,vu;function ib(){return vu||(vu=1,(function(s){(function(e,t,n){function r(l){var u=this,c=a();u.next=function(){var h=2091639*u.s0+u.c*23283064365386963e-26;return u.s0=u.s1,u.s1=u.s2,u.s2=h-(u.c=h|0)},u.c=1,u.s0=c(" "),u.s1=c(" "),u.s2=c(" "),u.s0-=c(l),u.s0<0&&(u.s0+=1),u.s1-=c(l),u.s1<0&&(u.s1+=1),u.s2-=c(l),u.s2<0&&(u.s2+=1),c=null}function i(l,u){return u.c=l.c,u.s0=l.s0,u.s1=l.s1,u.s2=l.s2,u}function o(l,u){var c=new r(l),h=u&&u.state,d=c.next;return d.int32=function(){return c.next()*4294967296|0},d.double=function(){return d()+(d()*2097152|0)*11102230246251565e-32},d.quick=d,h&&(typeof h=="object"&&i(h,c),d.state=function(){return i(c,{})}),d}function a(){var l=4022871197,u=function(c){c=String(c);for(var h=0;h<c.length;h++){l+=c.charCodeAt(h);var d=.02519603282416938*l;l=d>>>0,d-=l,d*=l,l=d>>>0,d-=l,l+=d*4294967296}return(l>>>0)*23283064365386963e-26};return u}t&&t.exports?t.exports=o:this.alea=o})(rb,s)})(mi)),mi.exports}var gi={exports:{}},ob=gi.exports,_u;function ab(){return _u||(_u=1,(function(s){(function(e,t,n){function r(a){var l=this,u="";l.x=0,l.y=0,l.z=0,l.w=0,l.next=function(){var h=l.x^l.x<<11;return l.x=l.y,l.y=l.z,l.z=l.w,l.w^=l.w>>>19^h^h>>>8},a===(a|0)?l.x=a:u+=a;for(var c=0;c<u.length+64;c++)l.x^=u.charCodeAt(c)|0,l.next()}function i(a,l){return l.x=a.x,l.y=a.y,l.z=a.z,l.w=a.w,l}function o(a,l){var u=new r(a),c=l&&l.state,h=function(){return(u.next()>>>0)/4294967296};return h.double=function(){do var d=u.next()>>>11,w=(u.next()>>>0)/4294967296,I=(d+w)/(1<<21);while(I===0);return I},h.int32=u.next,h.quick=h,c&&(typeof c=="object"&&i(c,u),h.state=function(){return i(u,{})}),h}t&&t.exports?t.exports=o:this.xor128=o})(ob,s)})(gi)),gi.exports}var yi={exports:{}},lb=yi.exports,Su;function ub(){return Su||(Su=1,(function(s){(function(e,t,n){function r(a){var l=this,u="";l.next=function(){var h=l.x^l.x>>>2;return l.x=l.y,l.y=l.z,l.z=l.w,l.w=l.v,(l.d=l.d+362437|0)+(l.v=l.v^l.v<<4^(h^h<<1))|0},l.x=0,l.y=0,l.z=0,l.w=0,l.v=0,a===(a|0)?l.x=a:u+=a;for(var c=0;c<u.length+64;c++)l.x^=u.charCodeAt(c)|0,c==u.length&&(l.d=l.x<<10^l.x>>>4),l.next()}function i(a,l){return l.x=a.x,l.y=a.y,l.z=a.z,l.w=a.w,l.v=a.v,l.d=a.d,l}function o(a,l){var u=new r(a),c=l&&l.state,h=function(){return(u.next()>>>0)/4294967296};return h.double=function(){do var d=u.next()>>>11,w=(u.next()>>>0)/4294967296,I=(d+w)/(1<<21);while(I===0);return I},h.int32=u.next,h.quick=h,c&&(typeof c=="object"&&i(c,u),h.state=function(){return i(u,{})}),h}t&&t.exports?t.exports=o:this.xorwow=o})(lb,s)})(yi)),yi.exports}var bi={exports:{}},cb=bi.exports,Iu;function hb(){return Iu||(Iu=1,(function(s){(function(e,t,n){function r(a){var l=this;l.next=function(){var c=l.x,h=l.i,d,w;return d=c[h],d^=d>>>7,w=d^d<<24,d=c[h+1&7],w^=d^d>>>10,d=c[h+3&7],w^=d^d>>>3,d=c[h+4&7],w^=d^d<<7,d=c[h+7&7],d=d^d<<13,w^=d^d<<9,c[h]=w,l.i=h+1&7,w};function u(c,h){var d,w=[];if(h===(h|0))w[0]=h;else for(h=""+h,d=0;d<h.length;++d)w[d&7]=w[d&7]<<15^h.charCodeAt(d)+w[d+1&7]<<13;for(;w.length<8;)w.push(0);for(d=0;d<8&&w[d]===0;++d);for(d==8?w[7]=-1:w[d],c.x=w,c.i=0,d=256;d>0;--d)c.next()}u(l,a)}function i(a,l){return l.x=a.x.slice(),l.i=a.i,l}function o(a,l){a==null&&(a=+new Date);var u=new r(a),c=l&&l.state,h=function(){return(u.next()>>>0)/4294967296};return h.double=function(){do var d=u.next()>>>11,w=(u.next()>>>0)/4294967296,I=(d+w)/(1<<21);while(I===0);return I},h.int32=u.next,h.quick=h,c&&(c.x&&i(c,u),h.state=function(){return i(u,{})}),h}t&&t.exports?t.exports=o:this.xorshift7=o})(cb,s)})(bi)),bi.exports}var wi={exports:{}},fb=wi.exports,ku;function db(){return ku||(ku=1,(function(s){(function(e,t,n){function r(a){var l=this;l.next=function(){var c=l.w,h=l.X,d=l.i,w,I;return l.w=c=c+1640531527|0,I=h[d+34&127],w=h[d=d+1&127],I^=I<<13,w^=w<<17,I^=I>>>15,w^=w>>>12,I=h[d]=I^w,l.i=d,I+(c^c>>>16)|0};function u(c,h){var d,w,I,E,m,S=[],b=128;for(h===(h|0)?(w=h,h=null):(h=h+"\0",w=0,b=Math.max(b,h.length)),I=0,E=-32;E<b;++E)h&&(w^=h.charCodeAt((E+32)%h.length)),E===0&&(m=w),w^=w<<10,w^=w>>>15,w^=w<<4,w^=w>>>13,E>=0&&(m=m+1640531527|0,d=S[E&127]^=w+m,I=d==0?I+1:0);for(I>=128&&(S[(h&&h.length||0)&127]=-1),I=127,E=512;E>0;--E)w=S[I+34&127],d=S[I=I+1&127],w^=w<<13,d^=d<<17,w^=w>>>15,d^=d>>>12,S[I]=w^d;c.w=m,c.X=S,c.i=I}u(l,a)}function i(a,l){return l.i=a.i,l.w=a.w,l.X=a.X.slice(),l}function o(a,l){a==null&&(a=+new Date);var u=new r(a),c=l&&l.state,h=function(){return(u.next()>>>0)/4294967296};return h.double=function(){do var d=u.next()>>>11,w=(u.next()>>>0)/4294967296,I=(d+w)/(1<<21);while(I===0);return I},h.int32=u.next,h.quick=h,c&&(c.X&&i(c,u),h.state=function(){return i(u,{})}),h}t&&t.exports?t.exports=o:this.xor4096=o})(fb,s)})(wi)),wi.exports}var xi={exports:{}},pb=xi.exports,Tu;function mb(){return Tu||(Tu=1,(function(s){(function(e,t,n){function r(a){var l=this,u="";l.next=function(){var h=l.b,d=l.c,w=l.d,I=l.a;return h=h<<25^h>>>7^d,d=d-w|0,w=w<<24^w>>>8^I,I=I-h|0,l.b=h=h<<20^h>>>12^d,l.c=d=d-w|0,l.d=w<<16^d>>>16^I,l.a=I-h|0},l.a=0,l.b=0,l.c=-1640531527,l.d=1367130551,a===Math.floor(a)?(l.a=a/4294967296|0,l.b=a|0):u+=a;for(var c=0;c<u.length+20;c++)l.b^=u.charCodeAt(c)|0,l.next()}function i(a,l){return l.a=a.a,l.b=a.b,l.c=a.c,l.d=a.d,l}function o(a,l){var u=new r(a),c=l&&l.state,h=function(){return(u.next()>>>0)/4294967296};return h.double=function(){do var d=u.next()>>>11,w=(u.next()>>>0)/4294967296,I=(d+w)/(1<<21);while(I===0);return I},h.int32=u.next,h.quick=h,c&&(typeof c=="object"&&i(c,u),h.state=function(){return i(u,{})}),h}t&&t.exports?t.exports=o:this.tychei=o})(pb,s)})(xi)),xi.exports}var vi={exports:{}},gb={},yb=Object.freeze({__proto__:null,default:gb}),bb=oy(yb),wb=vi.exports,Eu;function xb(){return Eu||(Eu=1,(function(s){(function(e,t,n){var r=256,i=6,o=52,a="random",l=n.pow(r,i),u=n.pow(2,o),c=u*2,h=r-1,d;function w(_,v,T){var N=[];v=v==!0?{entropy:!0}:v||{};var O=S(m(v.entropy?[_,f(t)]:_??b(),3),N),$=new I(N),A=function(){for(var g=$.g(i),p=l,y=0;g<u;)g=(g+y)*r,p*=r,y=$.g(1);for(;g>=c;)g/=2,p/=2,y>>>=1;return(g+y)/p};return A.int32=function(){return $.g(4)|0},A.quick=function(){return $.g(4)/4294967296},A.double=A,S(f($.S),t),(v.pass||T||function(g,p,y,x){return x&&(x.S&&E(x,$),g.state=function(){return E($,{})}),y?(n[a]=g,p):g})(A,O,"global"in v?v.global:this==n,v.state)}function I(_){var v,T=_.length,N=this,O=0,$=N.i=N.j=0,A=N.S=[];for(T||(_=[T++]);O<r;)A[O]=O++;for(O=0;O<r;O++)A[O]=A[$=h&$+_[O%T]+(v=A[O])],A[$]=v;(N.g=function(g){for(var p,y=0,x=N.i,k=N.j,C=N.S;g--;)p=C[x=h&x+1],y=y*r+C[h&(C[x]=C[k=h&k+p])+(C[k]=p)];return N.i=x,N.j=k,y})(r)}function E(_,v){return v.i=_.i,v.j=_.j,v.S=_.S.slice(),v}function m(_,v){var T=[],N=typeof _,O;if(v&&N=="object")for(O in _)try{T.push(m(_[O],v-1))}catch{}return T.length?T:N=="string"?_:_+"\0"}function S(_,v){for(var T=_+"",N,O=0;O<T.length;)v[h&O]=h&(N^=v[h&O]*19)+T.charCodeAt(O++);return f(v)}function b(){try{var _;return d&&(_=d.randomBytes)?_=_(r):(_=new Uint8Array(r),(e.crypto||e.msCrypto).getRandomValues(_)),f(_)}catch{var v=e.navigator,T=v&&v.plugins;return[+new Date,e,T,e.screen,f(t)]}}function f(_){return String.fromCharCode.apply(0,_)}if(S(n.random(),t),s.exports){s.exports=w;try{d=bb}catch{}}else n["seed"+a]=w})(typeof self<"u"?self:wb,[],Math)})(vi)),vi.exports}var Uo,Au;function vb(){if(Au)return Uo;Au=1;var s=ib(),e=ab(),t=ub(),n=hb(),r=db(),i=mb(),o=xb();return o.alea=s,o.xor128=e,o.xorwow=t,o.xorshift7=n,o.xor4096=r,o.tychei=i,Uo=o,Uo}var Jh=vb();/**
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
 */class ef{constructor(e,t,n,r,i){this.mean=e,this.stdDev=t,this.dtype=n,this.nextVal=NaN,this.truncated=r,this.truncated&&(this.upper=this.mean+this.stdDev*2,this.lower=this.mean-this.stdDev*2);const o=i||Math.random();this.random=Jh.alea(o.toString())}nextValue(){if(!isNaN(this.nextVal)){const r=this.nextVal;return this.nextVal=NaN,r}let e,t,n=!1;for(;!n;){let r,i,o;do r=2*this.random()-1,i=2*this.random()-1,o=r*r+i*i;while(o>=1||o===0);const a=Math.sqrt(-2*Math.log(o)/o);e=this.mean+this.stdDev*r*a,t=this.mean+this.stdDev*i*a,(!this.truncated||this.isValidTruncated(e))&&(n=!0)}return(!this.truncated||this.isValidTruncated(t))&&(this.nextVal=this.convertValue(t)),this.convertValue(e)}convertValue(e){return this.dtype==null||this.dtype==="float32"?e:Math.round(e)}isValidTruncated(e){return e<=this.upper&&e>=this.lower}}class _b{constructor(e=0,t=1,n,r){if(this.canReturnFloat=()=>this.dtype==null||this.dtype==="float32",this.min=e,this.range=t-e,this.dtype=n,r==null&&(r=Math.random()),typeof r=="number"&&(r=r.toString()),!this.canReturnFloat()&&this.range<=1)throw new Error(`The difference between ${e} - ${t} <= 1 and dtype is not float`);this.random=Jh.alea(r)}convertValue(e){return this.canReturnFloat()?e:Math.round(e)}nextValue(){return this.convertValue(this.min+this.range*this.random())}}/**
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
 */function Sb(s,e=0,t=1,n,r){if(Nn(s),n!=null&&n==="bool")throw new Error(`Unsupported data type ${n}`);const i=new ef(e,t,n,!1,r),o=Ye(s,n);for(let a=0;a<o.values.length;a++)o.values[a]=i.nextValue();return o.toTensor()}const Ib=Q({randomNormal_:Sb});/**
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
 */function kb(s,e=0,t=1,n="float32",r){Nn(s);const i=Ye(s,n),o=new _b(e,t,null,r);for(let a=0;a<i.values.length;a++)i.values[a]=o.nextValue();return i.toTensor()}const tf=Q({randomUniform_:kb});/**
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
 */function Pi(s,e,t=1,n="float32"){if(t===0)throw new Error("Cannot have a step of zero");const r={start:s,stop:e,step:t,dtype:n};return H.runKernel(Pg,{},r)}/**
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
 */function Tb(s){const t={input:q(s,"input","real")};return H.runKernel(Rg,t)}const Eb=Q({real_:Tb});/**
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
 */function Ab(s){const t={x:q(s,"x","relu")};return H.runKernel(Lg,t)}const Nr=Q({relu_:Ab});/**
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
 */function Cb(s){const t={x:q(s,"x","relu6")};return H.runKernel(Ug,t)}const $b=Q({relu6_:Cb});/**
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
 */function Nb(s){const t={x:q(s,"x","round")};return H.runKernel(zg,t)}const Db=Q({round_:Nb});/**
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
 */function Ob(s){const t={x:q(s,"x","selu")};return H.runKernel(Gg,t)}const Mb=Q({selu_:Ob});function Pb(s,e,t,n,r,i=[1,1],o="NHWC"){const a=q(s,"x","separableConv2d"),l=q(e,"depthwiseFilter","separableConv2d"),u=q(t,"pointwiseFilter","separableConv2d");let c=a,h=!1;if(a.rank===3&&(h=!0,c=se(a,[1,a.shape[0],a.shape[1],a.shape[2]])),o==="NCHW")throw new Error("separableConv2d currently does not support dataFormat NCHW; only NHWC is supported");P(c.rank===4,()=>`Error in separableConv2d: input must be rank 4, but got rank ${c.rank}.`),P(l.rank===4,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${l.rank}.`),P(u.rank===4,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${l.rank}.`),P(u.shape[0]===1,()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${u.shape[0]}.`),P(u.shape[1]===1,()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${u.shape[1]}.`);const d=l.shape[2],w=l.shape[3];P(u.shape[2]===d*w,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${d*w}, but got ${u.shape[2]}.`);const I=J0(c,l,n,r,o,i),m=Ya(I,u,1,"valid",o);return h?se(m,[m.shape[1],m.shape[2],m.shape[3]]):m}const Rb=Q({separableConv2d_:Pb});/**
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
 */function Lb(s,e,t){const n=q(s,"x","slice1d");return P(n.rank===1,()=>`slice1d expects a rank-1 tensor, but got a rank-${n.rank} tensor`),Je(n,[e],[t])}const el=Q({slice1d_:Lb});/**
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
 */function Bb(s,e,t){const n=q(s,"x","slice2d");return P(n.rank===2,()=>`slice2d expects a rank-2 tensor, but got a rank-${n.rank} tensor`),Je(n,e,t)}const nf=Q({slice2d_:Bb});/**
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
 */function Fb(s,e,t){const n=q(s,"x","slice3d");return P(n.rank===3,()=>`slice3d expects a rank-3 tensor, but got a rank-${n.rank} tensor`),Je(n,e,t)}const tl=Q({slice3d_:Fb});/**
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
 */function Ub(s,e=-1){const t=q(s,"logits","softmax","float32");if(e===-1&&(e=t.rank-1),e!==t.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${t.rank} and dim was ${e}`);const n={logits:t},r={dim:e};return H.runKernel(Xg,n,r)}const sf=Q({softmax_:Ub});/**
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
 */function zb(s,e,t=0){const r={x:q(s,"x","split")},i={numOrSizeSplits:e,axis:t};return H.runKernel(Kg,r,i)}const rf=Q({split_:zb});/**
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
 */function Vb(s,e){const t=q(s,"x","squeeze","string_or_numeric");return se(t,Sm(t.shape,e).newShape)}const yo=Q({squeeze_:Vb});/**
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
 */function Gb(s,e=0){const t=Rh(s,"tensors","stack","string_or_numeric");P(t.length>=1,()=>"Pass at least one tensor to tf.stack"),t.length>0&&P(e<=t[0].rank,()=>"Axis must be <= rank of the tensor");const n=t,r={axis:e};return H.runKernel(Dg,n,r)}const Ri=Q({stack_:Gb});/**
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
 */function Wb(s,e=0){const n={x:q(s,"x","step")},r={alpha:e};return H.runKernel(ty,n,r)}const qb=Q({step_:Wb});/**
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
 */function zo(s,e,t){if(uh(s),e!=null&&e.length!==2)throw new Error("tensor2d() requires shape to have two numbers");const n=fo(s,t);if(n.length!==2&&n.length!==1)throw new Error("tensor2d() requires values to be number[][] or flat/TypedArray");if(n.length===1&&e==null)throw new Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return po(s,e,n,t)}/**
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
 */function Hb(s,e=0,t=1,n,r){if(Nn(s),n!=null&&n==="bool")throw new Error("Unsupported data type $ { dtype }");const i=new ef(e,t,n,!0,r),o=Ye(s,n);for(let a=0;a<o.values.length;a++)o.values[a]=i.nextValue();return o.toTensor()}const of=Q({truncatedNormal_:Hb});/**
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
 */function jb(s,e=0){const t=q(s,"x","unstack","string_or_numeric");P(e>=-t.shape.length&&e<t.shape.length,()=>`Axis = ${e} is not in [-${t.shape.length}, ${t.shape.length})`);const n={value:t},r={axis:e};return H.runKernel(Jg,n,r)}const af=Q({unstack_:jb});/**
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
 */function Kb(s,e=!0,t,n){return H.makeVariable(s,e,t,n)}/**
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
 */function Xb(s,e,t){const n=q(s,"x","transpose");if(e==null&&(e=n.shape.map((o,a)=>a).reverse()),P(n.rank===e.length,()=>`Error in transpose: rank of input ${n.rank} must match length of perm ${e}.`),e.forEach(o=>{P(o>=0&&o<n.rank,()=>`All entries in 'perm' must be between 0 and ${n.rank-1} but got ${e}`)}),n.rank<=1)return n.clone();const r={x:n},i={perm:e};return n.dtype==="complex64"?Y(()=>{let o=Eb(n),a=N1(n);return o=H.runKernel(Mo,{x:o},i),a=H.runKernel(Mo,{x:a},i),t&&(a=Us(a)),qa(o,a)}):H.runKernel(Mo,r,i)}const Be=Q({transpose_:Xb});/**
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
 */function Yb(s,e,t,n,r,i="NHWC",o){let a=s;s.rank===3&&(a=se(s,[1,s.shape[0],s.shape[1],s.shape[2]]));let l=e;l.rank===3&&(l=se(e,[1,e.shape[0],e.shape[1],e.shape[2]])),P(a.rank===4,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${a.shape}.`),P(l.rank===4,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${l.shape}.`),P(t.length===4,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${t}.`);const u=i==="NHWC"?a.shape[3]:a.shape[1],c=i==="NHWC"?l.shape[3]:l.shape[1];P(u===t[2],()=>`Error in conv2dDerFilter: depth of input ${u}) must match input depth in filter (${t[2]}.`),P(c===t[3],()=>`Error in conv2dDerFilter: depth of dy (${c}) must match output depth for filter (${t[3]}).`),vn("conv2dDerFilter",r,o);const h={x:a,dy:l},d={strides:n,pad:r,dataFormat:i,dimRoundingMode:o,filterShape:t};return H.runKernel(qm,h,d)}const Qb=Q({conv2DBackpropFilter_:Yb});/**
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
 */function Zb(s,e,t){if(t==null||t==="linear")return s;if(t==="relu")return J(s,qb(e));throw new Error(`Cannot compute gradient for fused activation ${t}.`)}function Jb(s,e){let t=e;const n=e1(s.shape,e.shape);return n.length>0&&(t=Se(t,n)),se(t,s.shape)}function ew(s,e,t,n){if(e==="linear")return s;if(e==="relu")return Nr(s);if(e==="elu")return qh(s);if(e==="relu6")return $b(s);if(e==="prelu")return sb(s,t);if(e==="leakyrelu")return O1(s,n);if(e==="sigmoid")return Ka(s);throw new Error(`Unknown fused activation ${e}.`)}const tw=(s,e)=>!(s>0)||e==="linear";/**
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
 */function nw({x:s,filter:e,strides:t,pad:n,dataFormat:r="NHWC",dilations:i=[1,1],dimRoundingMode:o,bias:a,activation:l="linear",preluActivationWeights:u,leakyreluAlpha:c}){if(l=l||"linear",tw(H.state.gradientDepth,l)===!1){P(r==="NHWC",()=>`Error in fused conv2d: got dataFormat of ${r} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let T=Ya(s,e,t,n,r,i,o);return a!=null&&(T=ae(T,a)),ew(T,l,u,c)}const h=q(s,"x","conv2d","float32"),d=q(e,"filter","conv2d","float32");let w=h,I=!1;h.rank===3&&(I=!0,w=se(h,[1,h.shape[0],h.shape[1],h.shape[2]])),P(w.rank===4,()=>`Error in fused conv2d: input must be rank 4, but got rank ${w.rank}.`),P(d.rank===4,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${d.rank}.`),vn("fused conv2d",n,o);const E=r==="NHWC"?w.shape[3]:w.shape[1];P(d.shape[2]===E,()=>`Error in conv2d: depth of input (${E}) must match input depth for filter ${d.shape[2]}.`),P(Fs(t,i),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${t} and dilations '${i}'`);const m=ja(w.shape,d.shape,t,i,n,o);let S;a!=null&&(S=q(a,"bias","fused conv2d"),[S]=bt(S,h),r==="NHWC"?St(m.outShape,S.shape):(P(S.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${S.shape.length}.`),P(S.shape.length===0||S.shape[0]===m.outChannels||S.shape[0]===1,()=>`Error in fused conv2d: bias shape (${S.shape}) is not compatible with the number of output channels (${m.outChannels})`)));let b;if(u!=null){const T=u.shape;if(P(T.length<=1||T.length===3,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${T.length}.`),T.length===1)P(T[0]===1||T[0]===m.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${T}) is not compatible with the number of output channels (${m.outChannels}).`);else if(T.length===3)try{St(T,m.outShape)}catch{const O=`Error in fused conv2d: PReLU activation weights (${T}) is not compatible with the output shape of the conv2d (${m.outShape}).`;throw Error(O)}b=q(u,"prelu weights","fused conv2d")}const f=(T,N)=>{P(r==="NHWC",()=>`Error in gradient of fused conv2D: got dataFormat of ${r} but only NHWC is currently supported.`);const[O,$,A,g]=N,p=Zb(T,A,l);P(wa(i),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${i}'`);const y=Wh($.shape,p,O,t,n),x=Qb($,p,O.shape,t,n),k=[y,x];if(g!=null){const C=Jb(g,p);k.push(C)}return k},_={x:w,filter:d,bias:S,preluActivationWeights:b},v={strides:t,pad:n,dataFormat:r,dilations:i,dimRoundingMode:o,activation:l,leakyreluAlpha:c};return a==null?va((N,O,$)=>{let A=H.runKernel(ua,_,v);return $([O,N,A]),I&&(A=se(A,[A.shape[1],A.shape[2],A.shape[3]])),{value:A,gradFunc:f}})(w,d):va((N,O,$,A)=>{let g=H.runKernel(ua,_,v);return A([O,N,g,$]),I&&(g=se(g,[g.shape[1],g.shape[2],g.shape[3]])),{value:g,gradFunc:f}})(w,d,S)}const sw=Q({fusedConv2d_:nw});/**
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
 */function rw(s,e,t,n,r="bilinear",i=0){const o=q(s,"image","cropAndResize"),a=q(e,"boxes","cropAndResize","float32"),l=q(t,"boxInd","cropAndResize","int32"),u=a.shape[0];P(o.rank===4,()=>`Error in cropAndResize: image must be rank 4,but got rank ${o.rank}.`),P(a.rank===2&&a.shape[1]===4,()=>`Error in cropAndResize: boxes must be have size [${u},4] but had shape ${a.shape}.`),P(l.rank===1&&l.shape[0]===u,()=>`Error in cropAndResize: boxInd must be have size [${u}] but had shape ${a.shape}.`),P(n.length===2,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${n.length}.`),P(n[0]>=1&&n[1]>=1,()=>`cropSize must be atleast [1,1], but was ${n}`),P(r==="bilinear"||r==="nearest",()=>`method must be bilinear or nearest, but was ${r}`);const c={image:o,boxes:a,boxInd:l},h={method:r,extrapolationValue:i,cropSize:n};return H.runKernel(Xm,c,h)}const iw=Q({cropAndResize_:rw});/**
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
 */function ow(s){const e=q(s,"image","flipLeftRight","float32");P(e.rank===4,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${e.rank}.`);const t={image:e};return H.runKernel(ig,t,{})}const aw=Q({flipLeftRight_:ow});/**
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
 */function lw(s){const e=q(s,"image","grayscaleToRGB"),t=e.rank-1,n=e.shape[t];P(e.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${e.rank}.`),P(n===1,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${n}.`);const r=new Array(e.rank);return r.fill(1,0,t),r[t]=3,pi(e,r)}const uw=Q({grayscaleToRGB_:lw});/**
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
 */function cw(s){const e=q(s,"image","RGBToGrayscale"),t=e.rank-1,n=e.shape[t];P(e.rank>=2,()=>`Error in RGBToGrayscale: images must be at least rank 2, but got rank ${e.rank}.`),P(n===3,()=>`Error in RGBToGrayscale: last dimension of an RGB image should be size 3, but got size ${n}.`);const r=e.dtype,i=Ee(e,"float32"),o=pt([.2989,.587,.114]);let a;switch(e.rank){case 2:a=Zs("ij,j->i",i,o);break;case 3:a=Zs("ijk,k->ij",i,o);break;case 4:a=Zs("ijkl,l->ijk",i,o);break;case 5:a=Zs("ijklm,m->ijkl",i,o);break;case 6:a=Zs("ijklmn,n->ijklm",i,o);break;default:throw new Error("Not a valid tensor rank.")}return a=dn(a,-1),Ee(a,r)}const hw=Q({rgbToGrayscale_:cw});/**
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
 */function fw(s,e,t=0,n=.5){const r=q(s,"image","rotateWithOffset","float32");P(r.rank===4,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${r.rank}.`);const i={image:r},o={radians:e,fillValue:t,center:n};return H.runKernel(ny,i,o)}const dw=Q({rotateWithOffset_:fw});/**
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
 */function Vs(s,e,t,n,r,i){n==null&&(n=.5),r==null&&(r=Number.NEGATIVE_INFINITY),i==null&&(i=0);const o=s.shape[0];return t=Math.min(t,o),P(0<=n&&n<=1,()=>`iouThreshold must be in [0, 1], but was '${n}'`),P(s.rank===2,()=>`boxes must be a 2D tensor, but was of rank '${s.rank}'`),P(s.shape[1]===4,()=>`boxes must have 4 columns, but 2nd dimension was ${s.shape[1]}`),P(e.rank===1,()=>"scores must be a 1D tensor"),P(e.shape[0]===o,()=>`scores has incompatible shape with boxes. Expected ${o}, but was ${e.shape[0]}`),P(0<=i&&i<=1,()=>`softNmsSigma must be in [0, 1], but was '${i}'`),{maxOutputSize:t,iouThreshold:n,scoreThreshold:r,softNmsSigma:i}}/**
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
 */function pw(s,e,t,n=.5,r=Number.NEGATIVE_INFINITY){const i=q(s,"boxes","nonMaxSuppression","float32"),o=q(e,"scores","nonMaxSuppression","float32"),a=Vs(i,o,t,n,r);t=a.maxOutputSize,n=a.iouThreshold,r=a.scoreThreshold;const l={maxOutputSize:t,iouThreshold:n,scoreThreshold:r};return H.runKernel(Eg,{boxes:i,scores:o},l)}const mw=Q({nonMaxSuppression_:pw});/**
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
 */function gw(s,e,t){const n=yw(s,e,t),r=n<0?-(n+1):n;s.splice(r,0,e)}function yw(s,e,t){return ww(s,e,t||bw)}function bw(s,e){return s>e?1:s<e?-1:0}function ww(s,e,t){let n=0,r=s.length,i=0,o=!1;for(;n<r;){i=n+(r-n>>>1);const a=t(e,s[i]);a>0?n=i+1:(r=i,o=!a)}return o?n:-n-1}/**
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
 */function xw(s,e,t,n,r){return nl(s,e,t,n,r,0)}function vw(s,e,t,n,r,i){return nl(s,e,t,n,r,0,!1,i,!0)}function _w(s,e,t,n,r,i){return nl(s,e,t,n,r,i,!0)}function nl(s,e,t,n,r,i,o=!1,a=!1,l=!1){const u=[];for(let m=0;m<e.length;m++)e[m]>r&&u.push({score:e[m],boxIndex:m,suppressBeginIndex:0});u.sort(Cu);const c=i>0?-.5/i:0,h=[],d=[];for(;h.length<t&&u.length>0;){const m=u.pop(),{score:S,boxIndex:b,suppressBeginIndex:f}=m;if(S<r)break;let _=!1;for(let v=h.length-1;v>=f;--v){const T=Sw(s,b,h[v]);if(T>=n){_=!0;break}if(m.score=m.score*Iw(n,c,T),m.score<=r)break}m.suppressBeginIndex=h.length,_||(m.score===S?(h.push(b),d.push(m.score)):m.score>r&&gw(u,m,Cu))}const w=h.length,I=t-w;a&&I>0&&(h.push(...new Array(I).fill(0)),d.push(...new Array(I).fill(0)));const E={selectedIndices:h};return o&&(E.selectedScores=d),l&&(E.validOutputs=w),E}function Sw(s,e,t){const n=s.subarray(e*4,e*4+4),r=s.subarray(t*4,t*4+4),i=Math.min(n[0],n[2]),o=Math.min(n[1],n[3]),a=Math.max(n[0],n[2]),l=Math.max(n[1],n[3]),u=Math.min(r[0],r[2]),c=Math.min(r[1],r[3]),h=Math.max(r[0],r[2]),d=Math.max(r[1],r[3]),w=(a-i)*(l-o),I=(h-u)*(d-c);if(w<=0||I<=0)return 0;const E=Math.max(i,u),m=Math.max(o,c),S=Math.min(a,h),b=Math.min(l,d),f=Math.max(S-E,0)*Math.max(b-m,0);return f/(w+I-f)}function Iw(s,e,t){const n=Math.exp(e*t*t);return t<=s?n:0}function Cu(s,e){return s.score-e.score||s.score===e.score&&e.boxIndex-s.boxIndex}/**
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
 */async function kw(s,e,t,n=.5,r=Number.NEGATIVE_INFINITY){const i=q(s,"boxes","nonMaxSuppressionAsync"),o=q(e,"scores","nonMaxSuppressionAsync"),a=Vs(i,o,t,n,r);t=a.maxOutputSize,n=a.iouThreshold,r=a.scoreThreshold;const l=await Promise.all([i.data(),o.data()]),u=l[0],c=l[1],{selectedIndices:h}=xw(u,c,t,n,r);return i!==s&&i.dispose(),o!==e&&o.dispose(),pt(h,"int32")}const Tw=kw;/**
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
 */function Ew(s,e,t,n=.5,r=Number.NEGATIVE_INFINITY,i=0){const o=q(s,"boxes","nonMaxSuppression"),a=q(e,"scores","nonMaxSuppression"),l=Vs(o,a,t,n,r,i);t=l.maxOutputSize,n=l.iouThreshold,r=l.scoreThreshold,i=l.softNmsSigma;const u={boxes:o,scores:a},c={maxOutputSize:t,iouThreshold:n,scoreThreshold:r,softNmsSigma:i},h=H.runKernel(Cg,u,c);return{selectedIndices:h[0],selectedScores:h[1]}}const Aw=Q({nonMaxSuppressionWithScore_:Ew});/**
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
 */async function Cw(s,e,t,n=.5,r=Number.NEGATIVE_INFINITY,i=0){const o=q(s,"boxes","nonMaxSuppressionAsync"),a=q(e,"scores","nonMaxSuppressionAsync"),l=Vs(o,a,t,n,r,i);t=l.maxOutputSize,n=l.iouThreshold,r=l.scoreThreshold,i=l.softNmsSigma;const u=await Promise.all([o.data(),a.data()]),c=u[0],h=u[1],{selectedIndices:d,selectedScores:w}=_w(c,h,t,n,r,i);return o!==s&&o.dispose(),a!==e&&a.dispose(),{selectedIndices:pt(d,"int32"),selectedScores:pt(w)}}const $w=Cw;/**
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
 */function Nw(s,e,t,n=.5,r=Number.NEGATIVE_INFINITY,i=!1){const o=q(s,"boxes","nonMaxSuppression"),a=q(e,"scores","nonMaxSuppression"),l=Vs(o,a,t,n,r,null),u=l.maxOutputSize,c=l.iouThreshold,h=l.scoreThreshold,d={boxes:o,scores:a},w={maxOutputSize:u,iouThreshold:c,scoreThreshold:h,padToMaxOutputSize:i},I=H.runKernel(Ag,d,w);return{selectedIndices:I[0],validOutputs:I[1]}}const Dw=Q({nonMaxSuppressionPadded_:Nw});/**
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
 */async function Ow(s,e,t,n=.5,r=Number.NEGATIVE_INFINITY,i=!1){const o=q(s,"boxes","nonMaxSuppressionAsync"),a=q(e,"scores","nonMaxSuppressionAsync"),l=Vs(o,a,t,n,r,null),u=l.maxOutputSize,c=l.iouThreshold,h=l.scoreThreshold,[d,w]=await Promise.all([o.data(),a.data()]),{selectedIndices:I,validOutputs:E}=vw(d,w,u,c,h,i);return o!==s&&o.dispose(),a!==e&&a.dispose(),{selectedIndices:pt(I,"int32"),validOutputs:Rt(E,"int32")}}const Mw=Ow;/**
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
 */function Pw(s,e,t=!1,n=!1){const r=q(s,"images","resizeBilinear");P(r.rank===3||r.rank===4,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${r.rank}.`),P(e.length===2,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${e}.`),P(n===!1||t===!1,()=>"Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.");let i=r,o=!1;r.rank===3&&(o=!0,i=se(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const a={images:i},l={alignCorners:t,halfPixelCenters:n,size:e},u=H.runKernel(Fg,a,l);return o?se(u,[u.shape[1],u.shape[2],u.shape[3]]):u}const Rw=Q({resizeBilinear_:Pw});/**
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
 */function Lw(s,e,t=!1,n=!1){const r=q(s,"images","resizeNearestNeighbor");P(r.rank===3||r.rank===4,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${r.rank}.`),P(e.length===2,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${e}.`),P(r.dtype==="float32"||r.dtype==="int32",()=>"`images` must have `int32` or `float32` as dtype"),P(n===!1||t===!1,()=>"Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.");let i=r,o=!1;r.rank===3&&(o=!0,i=se(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const a={images:i},l={alignCorners:t,halfPixelCenters:n,size:e},u=H.runKernel(wh,a,l);return o?se(u,[u.shape[1],u.shape[2],u.shape[3]]):u}const Bw=Q({resizeNearestNeighbor_:Lw});/**
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
 */function Fw(s,e="binary",t=!1,n=.5){const r=q(s,"image","threshold"),i=.2989,o=.587,a=.114,l=r.shape[0]*r.shape[1];let u=J(pt([n]),255),c,h,d,w;if(P(r.rank===3,()=>`Error in threshold: image must be rank 3,but got rank ${r.rank}.`),P(r.shape[2]===3||r.shape[2]===1,()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${r.shape[2]}.`),P(r.dtype==="int32"||r.dtype==="float32",()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${r.dtype}.`),P(e==="otsu"||e==="binary",()=>`Method must be binary or otsu, but was ${e}`),r.shape[2]===3){[c,h,d]=rf(r,[1,1,1],-1);const m=J(c,i),S=J(h,o),b=J(d,a);w=ae(ae(m,S),b)}else w=s;if(e==="otsu"){const m=L0(Ee(Db(w),"int32"),fi([]),256);u=Uw(m,l)}const I=t?Qh(w,u):$r(w,u);return Ee(J(I,255),"int32")}function Uw(s,e){let t=pt([-1]),n=pt([0]),r=pt([0]),i,o,a,l,u,c;for(let h=0;h<s.size-1;h++){i=Je(s,0,h+1),o=Je(s,h+1),u=ge(Se(i),e),c=ge(Se(o),e);const d=Se(J(i,Pi(0,i.size)));a=ge(d,Se(i));const w=mo(o.shape,i.size),I=ae(Pi(0,o.size),w),E=J(o,I);l=ge(Se(E),Se(o));const m=we(a,l),S=we(a,l),b=J(u,c);r=J(J(b,m),S);const f=$r(r,n);n=ts(f,r,n),t=ts(f,pt([h]),t)}return t}const zw=Q({threshold_:Fw});/**
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
 */function Vw(s,e,t="nearest",n="constant",r=0,i){const o=q(s,"image","transform","float32"),a=q(e,"transforms","transform","float32");P(o.rank===4,()=>`Error in transform: image must be rank 4,but got rank ${o.rank}.`),P(a.rank===2&&(a.shape[0]===o.shape[0]||a.shape[0]===1)&&a.shape[1]===8,()=>"Error in transform: Input transform should be batch x 8 or 1 x 8"),P(i==null||i.length===2,()=>`Error in transform: outputShape must be [height, width] or null, but got ${i}.`);const l={image:o,transforms:a},u={interpolation:t,fillMode:n,fillValue:r,outputShape:i};return H.runKernel(Zg,l,u)}const Gw=Q({transform_:Vw});/**
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
 */function Ww(s,e,t){const n=q(s,"a","bandPart");P(n.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${n.rank}.`);const r=n.shape,[i,o]=n.shape.slice(-2);let a,l;typeof e=="number"?(P(e%1===0,()=>`bandPart(): numLower must be an integer, got ${e}.`),P(e<=i,()=>`bandPart(): numLower (${e}) must not be greater than the number of rows (${i}).`),a=q(e<0?i:e,"numLower","bandPart")):(P(e.dtype==="int32",()=>"bandPart(): numLower's dtype must be an int32."),a=ts(xu(e,0),i,Mi(e,i))),typeof t=="number"?(P(t%1===0,()=>`bandPart(): numUpper must be an integer, got ${t}.`),P(t<=o,()=>`bandPart(): numUpper (${t}) must not be greater than the number of columns (${o}).`),l=q(t<0?o:t,"numUpper","bandPart")):(P(t.dtype==="int32",()=>"bandPart(): numUpper's dtype must be an int32."),l=ts(xu(t,0),o,Mi(t,o)));const u=se(Pi(0,i,1,"int32"),[-1,1]),c=Pi(0,o,1,"int32"),h=we(u,c),d=go(Qh(h,a),C1(h,Us(l))),w=Os([i,o],n.dtype);return se(Ri(af(se(n,[-1,i,o])).map(I=>ts(d,I,w))),r)}const qw=Q({bandPart_:Ww});/**
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
 */function Hw(s){let e;if(Array.isArray(s)){e=!1,P(s!=null&&s.length>0,()=>"Gram-Schmidt process: input must not be null, undefined, or empty");const r=s[0].shape[0];for(let i=1;i<s.length;++i)P(s[i].shape[0]===r,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${s[i].shape[0]} vs. ${r})`)}else e=!0,s=rf(s,s.shape[0],0).map(r=>yo(r,[0]));P(s.length<=s[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${s.length}) exceeds number of dimensions (${s[0].shape[0]}).`);const t=[],n=s;for(let r=0;r<s.length;++r)t.push(H.tidy(()=>{let i=n[r];if(r>0)for(let o=0;o<r;++o){const a=J(Se(J(t[o],i)),t[o]);i=we(i,a)}return ge(i,Xh(i,"euclidean"))}));return e?Ri(t,0):t}const jw=Q({gramSchmidt_:Hw});/**
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
 */function Kw(s,e=!1){if(P(s.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${s.rank}`),s.rank===2)return $u(s,e);{const t=s.shape.slice(0,s.shape.length-2).reduce((l,u)=>l*u),n=af(se(s,[t,s.shape[s.shape.length-2],s.shape[s.shape.length-1]]),0),r=[],i=[];n.forEach(l=>{const[u,c]=$u(l,e);r.push(u),i.push(c)});const o=se(Ri(r,0),s.shape),a=se(Ri(i,0),s.shape);return[o,a]}}function $u(s,e=!1){return H.tidy(()=>{P(s.shape.length===2,()=>`qr2d() requires a 2D Tensor, but got a ${s.shape.length}D Tensor.`);const t=s.shape[0],n=s.shape[1];let r=Yh(t),i=Jn(s);const o=zo([[1]],[1,1]);let a=Jn(o);const l=t>=n?n:t;for(let u=0;u<l;++u){const c=i,h=a,d=r;[a,i,r]=H.tidy(()=>{const w=Je(i,[u,u],[t-u,1]),I=Xh(w),E=Je(i,[u,u],[1,1]),m=ts($r(E,0),zo([[-1]]),zo([[1]])),S=we(E,J(m,I)),b=ge(w,S);b.shape[0]===1?a=Jn(o):a=es([o,Je(b,[1,0],[b.shape[0]-1,b.shape[1]])],0);const f=Us(ge(ln(m,S),I)),_=Je(i,[u,0],[t-u,n]),v=J(f,a),T=Be(a);if(u===0)i=we(_,ln(v,ln(T,_)));else{const $=we(_,ln(v,ln(T,_)));i=es([Je(i,[0,0],[u,n]),$],0)}const N=Be(v),O=Je(r,[0,u],[t,r.shape[1]-u]);if(u===0)r=we(O,ln(ln(O,a),N));else{const $=we(O,ln(ln(O,a),N));r=es([Je(r,[0,0],[t,u]),$],1)}return[a,i,r]}),Ce([c,h,d])}return!e&&t>n&&(r=Je(r,[0,0],[t,n]),i=Je(i,[0,0],[n,n])),[r,i]})}const Xw=Q({qr_:Kw});/**
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
 */const Xr={flipLeftRight:aw,grayscaleToRGB:uw,resizeNearestNeighbor:Bw,resizeBilinear:Rw,rgbToGrayscale:hw,rotateWithOffset:dw,cropAndResize:iw,nonMaxSuppression:mw,nonMaxSuppressionAsync:Tw,nonMaxSuppressionWithScore:Aw,nonMaxSuppressionWithScoreAsync:$w,nonMaxSuppressionPadded:Dw,nonMaxSuppressionPaddedAsync:Mw,threshold:zw,transform:Gw},Yw={bandPart:qw,gramSchmidt:jw,qr:Xw};/**
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
 */const Qw=new Map,Zw=new Map;class Gs{getClassName(){return this.constructor.className}static fromConfig(e,t){return new e(t)}}class Vt{constructor(){this.classNameMap={}}static getMap(){return Vt.instance==null&&(Vt.instance=new Vt),Vt.instance}static register(e){Vt.getMap().classNameMap[e.className]=[e,e.fromConfig]}}function re(s,e,t){P(s.className!=null,()=>"Class being registered does not have the static className property defined."),P(typeof s.className=="string",()=>"className is required to be a string, but got type "+typeof s.className),P(s.className.length>0,()=>"Class being registered has an empty-string as its className, which is disallowed."),typeof e>"u"&&(e="Custom"),typeof t>"u"&&(t=s.className);const n=t,r=e+">"+n;return Vt.register(s),Qw.set(r,s),Zw.set(s,r),s}/**
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
 */class qn extends Gs{minimize(e,t=!1,n){const{value:r,grads:i}=this.computeGradients(e,n);if(n!=null){const o=n.map(a=>({name:a.name,tensor:i[a.name]}));this.applyGradients(o)}else this.applyGradients(i);return Ce(i),t?r:(r.dispose(),null)}get iterations(){return this.iterations_==null&&(this.iterations_=0),this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(e,t){return F1(e,t)}dispose(){this.iterations_!=null&&Ce(this.iterations_)}async saveIterations(){return this.iterations_==null&&(this.iterations_=0),{name:"iter",tensor:Rt(this.iterations_,"int32")}}async getWeights(){throw new Error("getWeights() is not implemented for this optimizer yet.")}async setWeights(e){throw new Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(e){return this.iterations_=(await e[0].tensor.data())[0],e.slice(1)}}Object.defineProperty(qn,Symbol.hasInstance,{value:s=>s.minimize!=null&&s.computeGradients!=null&&s.applyGradients!=null});/**
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
 */class lf extends qn{static get className(){return"Adadelta"}constructor(e,t,n=null){super(),this.learningRate=e,this.rho=t,this.epsilon=n,this.accumulatedGrads=[],this.accumulatedUpdates=[],n==null&&(this.epsilon=H.backend.epsilon())}applyGradients(e){(Array.isArray(e)?e.map(n=>n.name):Object.keys(e)).forEach((n,r)=>{const i=H.registeredVariables[n],o=!1;this.accumulatedGrads[r]==null&&(this.accumulatedGrads[r]={originalName:`${n}/accum_grad`,variable:Y(()=>bn(i).variable(o))}),this.accumulatedUpdates[r]==null&&(this.accumulatedUpdates[r]={originalName:`${n}/accum_var`,variable:Y(()=>bn(i).variable(o))});const a=Array.isArray(e)?e[r].tensor:e[n];if(a==null)return;const l=this.accumulatedGrads[r].variable,u=this.accumulatedUpdates[r].variable;Y(()=>{const c=ae(J(l,this.rho),J(Un(a),1-this.rho)),h=J(ge(sn(ae(u,this.epsilon)),sn(ae(l,this.epsilon))),a),d=ae(J(u,this.rho),J(Un(h),1-this.rho));l.assign(c),u.assign(d);const w=ae(J(h,-this.learningRate),i);i.assign(w)})}),this.incrementIterations()}dispose(){this.accumulatedUpdates!=null&&(Ce(this.accumulatedGrads.map(e=>e.variable)),Ce(this.accumulatedUpdates.map(e=>e.variable)))}async getWeights(){const e=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(e.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(e){e=await this.extractIterations(e);const t=e.length/2,n=!1;this.accumulatedGrads=e.slice(0,t).map(r=>({originalName:r.name,variable:r.tensor.variable(n)})),this.accumulatedUpdates=e.slice(t,t*2).map(r=>({originalName:r.name,variable:r.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.rho,t.epsilon)}}/**
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
 */class uf extends qn{static get className(){return"Adagrad"}constructor(e,t=.1){super(),this.learningRate=e,this.initialAccumulatorValue=t,this.accumulatedGrads=[]}applyGradients(e){(Array.isArray(e)?e.map(n=>n.name):Object.keys(e)).forEach((n,r)=>{const i=H.registeredVariables[n];this.accumulatedGrads[r]==null&&(this.accumulatedGrads[r]={originalName:`${n}/accumulator`,variable:Y(()=>mo(i.shape,this.initialAccumulatorValue).variable(!1))});const o=Array.isArray(e)?e[r].tensor:e[n];if(o==null)return;const a=this.accumulatedGrads[r].variable;Y(()=>{const l=ae(a,Un(o));a.assign(l);const u=ae(J(ge(o,sn(ae(l,H.backend.epsilon()))),-this.learningRate),i);i.assign(u)})}),this.incrementIterations()}dispose(){this.accumulatedGrads!=null&&Ce(this.accumulatedGrads.map(e=>e.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);const t=!1;this.accumulatedGrads=e.map(n=>({originalName:n.name,variable:n.tensor.variable(t)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(e,t){return new e(t.learningRate,t.initialAccumulatorValue)}}/**
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
 */class cf extends qn{static get className(){return"Adam"}constructor(e,t,n,r=null){super(),this.learningRate=e,this.beta1=t,this.beta2=n,this.epsilon=r,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],Y(()=>{this.accBeta1=Rt(t).variable(),this.accBeta2=Rt(n).variable()}),r==null&&(this.epsilon=H.backend.epsilon())}applyGradients(e){const t=Array.isArray(e)?e.map(n=>n.name):Object.keys(e);Y(()=>{const n=we(1,this.accBeta1),r=we(1,this.accBeta2);t.forEach((i,o)=>{const a=H.registeredVariables[i],l=!1;this.accumulatedFirstMoment[o]==null&&(this.accumulatedFirstMoment[o]={originalName:`${i}/m`,variable:Y(()=>bn(a).variable(l))}),this.accumulatedSecondMoment[o]==null&&(this.accumulatedSecondMoment[o]={originalName:`${i}/v`,variable:Y(()=>bn(a).variable(l))});const u=Array.isArray(e)?e[o].tensor:e[i];if(u==null)return;const c=this.accumulatedFirstMoment[o].variable,h=this.accumulatedSecondMoment[o].variable,d=ae(J(c,this.beta1),J(u,1-this.beta1)),w=ae(J(h,this.beta2),J(Un(u),1-this.beta2)),I=ge(d,n),E=ge(w,r);c.assign(d),h.assign(w);const m=ae(J(ge(I,ae(sn(E),this.epsilon)),-this.learningRate),a);a.assign(m)}),this.accBeta1.assign(J(this.accBeta1,this.beta1)),this.accBeta2.assign(J(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),this.accumulatedFirstMoment!=null&&Ce(this.accumulatedFirstMoment.map(e=>e.variable)),this.accumulatedSecondMoment!=null&&Ce(this.accumulatedSecondMoment.map(e=>e.variable))}async getWeights(){const e=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(e.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(e){e=await this.extractIterations(e),Y(()=>{this.accBeta1.assign(Oi(this.beta1,this.iterations_+1)),this.accBeta2.assign(Oi(this.beta2,this.iterations_+1))});const t=e.length/2,n=!1;this.accumulatedFirstMoment=e.slice(0,t).map(r=>({originalName:r.name,variable:r.tensor.variable(n)})),this.accumulatedSecondMoment=e.slice(t,t*2).map(r=>({originalName:r.name,variable:r.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon)}}/**
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
 */class hf extends qn{static get className(){return"Adamax"}constructor(e,t,n,r=null,i=0){super(),this.learningRate=e,this.beta1=t,this.beta2=n,this.epsilon=r,this.decay=i,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],Y(()=>{this.iteration=Rt(0).variable(),this.accBeta1=Rt(t).variable()}),r==null&&(this.epsilon=H.backend.epsilon())}applyGradients(e){const t=Array.isArray(e)?e.map(n=>n.name):Object.keys(e);Y(()=>{const n=we(1,this.accBeta1),r=ge(-this.learningRate,ae(J(this.iteration,this.decay),1));t.forEach((i,o)=>{const a=H.registeredVariables[i],l=!1;this.accumulatedFirstMoment[o]==null&&(this.accumulatedFirstMoment[o]={originalName:`${i}/m`,variable:bn(a).variable(l)}),this.accumulatedWeightedInfNorm[o]==null&&(this.accumulatedWeightedInfNorm[o]={originalName:`${i}/v`,variable:bn(a).variable(l)});const u=Array.isArray(e)?e[o].tensor:e[i];if(u==null)return;const c=this.accumulatedFirstMoment[o].variable,h=this.accumulatedWeightedInfNorm[o].variable,d=ae(J(c,this.beta1),J(u,1-this.beta1)),w=J(h,this.beta2),I=dt(u),E=zs(w,I);c.assign(d),h.assign(E);const m=ae(J(ge(r,n),ge(d,ae(E,this.epsilon))),a);a.assign(m)}),this.iteration.assign(ae(this.iteration,1)),this.accBeta1.assign(J(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),this.accumulatedFirstMoment!=null&&Ce(this.accumulatedFirstMoment.map(e=>e.variable)),this.accumulatedWeightedInfNorm!=null&&Ce(this.accumulatedWeightedInfNorm.map(e=>e.variable))}async getWeights(){throw new Error("getWeights() is not implemented for Adamax yet.")}async setWeights(e){throw new Error("setWeights() is not implemented for Adamax yet.")}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(e,t){return new e(t.learningRate,t.beta1,t.beta2,t.epsilon,t.decay)}}/**
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
 */class sl extends qn{static get className(){return"SGD"}constructor(e){super(),this.learningRate=e,this.setLearningRate(e)}applyGradients(e){(Array.isArray(e)?e.map(n=>n.name):Object.keys(e)).forEach((n,r)=>{const i=Array.isArray(e)?e[r].tensor:e[n];if(i==null)return;const o=H.registeredVariables[n];Y(()=>{const a=ae(J(this.c,i),o);o.assign(a)})}),this.incrementIterations()}setLearningRate(e){this.learningRate=e,this.c!=null&&this.c.dispose(),this.c=As(Rt(-e))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(e){if(e=await this.extractIterations(e),e.length!==0)throw new Error("SGD optimizer does not have settable weights.")}getConfig(){return{learningRate:this.learningRate}}static fromConfig(e,t){return new e(t.learningRate)}}/**
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
 */class ff extends sl{static get className(){return"Momentum"}constructor(e,t,n=!1){super(e),this.learningRate=e,this.momentum=t,this.useNesterov=n,this.accumulations=[],this.m=Rt(this.momentum)}applyGradients(e){(Array.isArray(e)?e.map(n=>n.name):Object.keys(e)).forEach((n,r)=>{const i=H.registeredVariables[n];this.accumulations[r]==null&&(this.accumulations[r]={originalName:`${n}/momentum`,variable:Y(()=>bn(i).variable(!1))});const o=this.accumulations[r].variable,a=Array.isArray(e)?e[r].tensor:e[n];a!=null&&Y(()=>{let l;const u=ae(J(this.m,o),a);this.useNesterov?l=ae(J(this.c,ae(a,J(u,this.m))),i):l=ae(J(this.c,u),i),o.assign(u),i.assign(l)})}),this.incrementIterations()}dispose(){this.m.dispose(),this.accumulations!=null&&Ce(this.accumulations.map(e=>e.variable))}setMomentum(e){this.momentum=e}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(e=>({name:e.originalName,tensor:e.variable})))}async setWeights(e){e=await this.extractIterations(e);const t=!1;this.accumulations=e.map(n=>({originalName:n.name,variable:n.tensor.variable(t)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(e,t){return new e(t.learningRate,t.momentum,t.useNesterov)}}/**
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
 */class df extends qn{static get className(){return"RMSProp"}constructor(e,t=.9,n=0,r=null,i=!1){if(super(),this.learningRate=e,this.decay=t,this.momentum=n,this.epsilon=r,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=i,r==null&&(this.epsilon=H.backend.epsilon()),e==null)throw new Error("learningRate for RMSPropOptimizer must be defined.")}applyGradients(e){(Array.isArray(e)?e.map(n=>n.name):Object.keys(e)).forEach((n,r)=>{const i=H.registeredVariables[n],o=!1;this.accumulatedMeanSquares[r]==null&&(this.accumulatedMeanSquares[r]={originalName:`${n}/rms`,variable:Y(()=>bn(i).variable(o))}),this.accumulatedMoments[r]==null&&(this.accumulatedMoments[r]={originalName:`${n}/momentum`,variable:Y(()=>bn(i).variable(o))}),this.accumulatedMeanGrads[r]==null&&this.centered&&(this.accumulatedMeanGrads[r]={originalName:`${n}/mg`,variable:Y(()=>bn(i).variable(o))});const a=Array.isArray(e)?e[r].tensor:e[n];if(a==null)return;const l=this.accumulatedMeanSquares[r].variable,u=this.accumulatedMoments[r].variable;Y(()=>{const c=ae(J(l,this.decay),J(Un(a),1-this.decay));if(this.centered){const h=this.accumulatedMeanGrads[r].variable,d=ae(J(h,this.decay),J(a,1-this.decay)),w=ge(J(a,this.learningRate),sn(we(c,ae(Un(d),this.epsilon)))),I=ae(J(u,this.momentum),w);l.assign(c),h.assign(d),u.assign(I);const E=we(i,I);i.assign(E)}else{const h=ae(J(l,this.decay),J(Un(a),1-this.decay)),d=ae(J(u,this.momentum),ge(J(a,this.learningRate),sn(ae(h,this.epsilon))));l.assign(h),u.assign(d);const w=we(i,d);i.assign(w)}})}),this.incrementIterations()}dispose(){this.accumulatedMeanSquares!=null&&Ce(this.accumulatedMeanSquares.map(e=>e.variable)),this.accumulatedMeanGrads!=null&&this.centered&&Ce(this.accumulatedMeanGrads.map(e=>e.variable)),this.accumulatedMoments!=null&&Ce(this.accumulatedMoments.map(e=>e.variable))}async getWeights(){const e=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&e.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(e.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(e){e=await this.extractIterations(e);const t=this.centered?e.length/3:e.length/2,n=!1;this.accumulatedMeanSquares=e.slice(0,t).map(r=>({originalName:r.name,variable:r.tensor.variable(n)})),this.accumulatedMoments=e.slice(t,t*2).map(r=>({originalName:r.name,variable:r.tensor.variable(n)})),this.centered&&(this.accumulatedMeanGrads=e.slice(t*2,t*3).map(r=>({originalName:r.name,variable:r.tensor.variable(n)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(e,t){return new e(t.learningRate,t.decay,t.momentum,t.epsilon,t.centered)}}/**
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
 */const Jw=[lf,uf,cf,hf,ff,df,sl];function ex(){for(const s of Jw)re(s)}/**
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
 */function tx(s,e,t){const n=s.shape.length;P(n===e.length,()=>`Error in slice${n}D: Length of begin ${e} must match the rank of the array (${n}).`),P(n===t.length,()=>`Error in slice${n}D: Length of size ${t} must match the rank of the array (${n}).`);for(let r=0;r<n;++r)P(e[r]+t[r]<=s.shape[r],()=>`Error in slice${n}D: begin[${r}] + size[${r}] (${e[r]+t[r]}) would overflow input.shape[${r}] (${s.shape[r]})`)}function nx(s,e,t){let n=t.length;for(let r=0;r<t.length;r++)if(t[r]>1){n=r;break}for(let r=n+1;r<t.length;r++)if(e[r]>0||t[r]!==s[r])return!1;return!0}function sx(s,e){let t=s.length>0?s[s.length-1]:1;for(let n=0;n<s.length-1;n++)t+=s[n]*e[n];return t}function rx(s,e,t){let n;const r=s.shape.length;typeof e=="number"?n=[e,...new Array(r-1).fill(0)]:e.length<r?n=e.concat(new Array(r-e.length).fill(0)):n=e.slice(),n.forEach(o=>{P(o!==-1,()=>"slice() does not support negative begin indexing.")});let i;return t==null?i=new Array(r).fill(-1):typeof t=="number"?i=[t,...new Array(r-1).fill(-1)]:t.length<r?i=t.concat(new Array(r-t.length).fill(-1)):i=t,i=i.map((o,a)=>o>=0?o:(P(o===-1,()=>`Negative size values should be exactly -1 but got ${o} for the slice() size at index ${a}.`),s.shape[a]-n[a])),[n,i]}/**
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
 */class ix{static sgd(e){return new sl(e)}static momentum(e,t,n=!1){return new ff(e,t,n)}static rmsprop(e,t=.9,n=0,r=null,i=!1){return new df(e,t,n,r,i)}static adam(e=.001,t=.9,n=.999,r=null){return new cf(e,t,n,r)}static adadelta(e=.001,t=.95,n=null){return new lf(e,t,n)}static adamax(e=.002,t=.9,n=.999,r=null,i=0){return new hf(e,t,n,r,i)}static adagrad(e,t=.1){return new uf(e,t)}}/**
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
 */const ys=ix;/**
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
 */const ox=typeof requestAnimationFrame<"u"?requestAnimationFrame:typeof setImmediate<"u"?setImmediate:s=>s();function ax(){return new Promise(s=>ox(()=>s()))}/**
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
 */function lx(s,e){const t=s[0].length;s.forEach((r,i)=>{P(r.length===t,()=>`Error in concat${t}D: rank of tensors[${i}] must be the same as the rank of the rest (${t})`)}),P(e>=0&&e<t,()=>`Error in concat${t}D: axis must be between 0 and ${t-1}.`);const n=s[0];s.forEach((r,i)=>{for(let o=0;o<t;o++)P(o===e||r[o]===n[o],()=>`Error in concat${t}D: Shape of tensors[${i}] (${r}) does not match the shape of the rest (${n}) along the non-concatenated axis ${i}.`)})}function wr(s,e){const t=s[0].slice();for(let n=1;n<s.length;n++)t[e]+=s[n][e];return t}/**
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
 */var pn;(function(s){s[s.FIRST_DIM_SIZE=0]="FIRST_DIM_SIZE",s[s.VALUE_ROWIDS=1]="VALUE_ROWIDS",s[s.ROW_LENGTHS=2]="ROW_LENGTHS",s[s.ROW_SPLITS=3]="ROW_SPLITS",s[s.ROW_LIMITS=4]="ROW_LIMITS",s[s.ROW_STARTS=5]="ROW_STARTS"})(pn||(pn={}));function ux(s,e,t){let n=new Array;if(t==null&&e==null)return n;if(e==null)for(;n.length<s+t.length;)n.push(-1);else n=e.slice();if(t==null)return n;if(s+t.length!==n.length)throw new Error(`rt input.shape and shape=${e} are incompatible: rt input.rank = ${s+t.length}, but shape.rank = ${n.length}`);for(let r=1;r<t.length;++r){const i=t[r],o=n[n.length-t.length+r],a=n[o];if(i>=0)if(a>=0){if(a!==i)throw new Error(`rt input.shape and shape=${e} are incompatible: rt input.shape[${r+s}] = ${i} but shape[${r+s}] = ${a}`)}else n[o]=i}return n}function cx(s){const e={FIRST_DIM_SIZE:pn.FIRST_DIM_SIZE,VALUE_ROWIDS:pn.VALUE_ROWIDS,ROW_LENGTHS:pn.ROW_LENGTHS,ROW_SPLITS:pn.ROW_SPLITS,ROW_LIMITS:pn.ROW_LIMITS,ROW_STARTS:pn.ROW_STARTS},t=[];for(const n of s)if(n in e)t.push(e[n]);else break;return t}function hx(s){return s.length===0?0:s[0]===pn.FIRST_DIM_SIZE?s.length-1:s.length}function fx(s,e){if(s==null||e==null)return;const t=s.length,n=e.length;if(t>=n)throw new Error(`defaultValue.shape=${s} and ragged tensor flatValues.shape=${e}, are incompatible: defaultValue.rank = ${t} must be less than ragged tensor input flatValues.rank = ${n})`);for(let r=0;r<Math.min(t,n-1);++r){const i=s[r],o=e[r+1];if(i>=0&&o>=0&&i!==1&&i!==o)throw new Error(`defaultValue.shape=${s}, and ragged tensor input flatValues.shape=${e} are incompatible: defaultValue.shape[${r-s.length}] = ${i} but ragged tensor input.flatValues.shape[${r-s.length}] = ${o}`)}}/**
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
 */const dx=1.7580993408473768,px=1.0507009873554805;/**
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
 */const mx=.3275911,gx=.254829592,yx=-.284496736,bx=1.421413741,wx=-1.453152027,xx=1.061405429;/**
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
 */function Nu(s,e){if(s.length!==e.length)throw new Error(`Cannot merge real and imag arrays of different lengths. real:${s.length}, imag: ${e.length}.`);const t=new Float32Array(s.length*2);for(let n=0;n<t.length;n+=2)t[n]=s[n/2],t[n+1]=e[n/2];return t}/**
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
 */function vx(s){return`Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${s}`}function _x(s,e){return`indices(${s}, 0) is invalid: ${e} < 0`}function Sx(s,e,t){return`indices(${s}, 0) is invalid: ${e} >= ${t}`}/**
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
 */function Ix(s,e){return`only one output dimension may be -1, not both ${s} and ${e}`}function kx(s,e){return`size ${s} must be non-negative, not ${e}`}function Tx(){return"reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero"}function Ex(s,e){const t=he(s),n=he(e);return`Input to reshape is a SparseTensor with ${t}
  dense values, but the requested shape requires a multiple of ${n}. inputShape=${s} outputShape= ${e}`}function Ax(s,e){const t=he(s),n=he(e);return`Input to reshape is a tensor with ${t} dense values, but the requested shape has ${n}. inputShape=${s} outputShape=${e}`}/**
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
 */function Du(){return"segment ids must be >= 0"}function Cx(){return"segment ids are not increasing"}function $x(s,e){return`Segment id ${s} out of range [0, ${e}), possibly because segmentIds input is not sorted.`}function Nx(s,e,t){return`Bad: indices[${s}] == ${e} out of range [0, ${t})`}/**
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
 */function pf(s){try{return s.map(e=>Ei(e))}catch(e){throw new Error(`Failed to decode encoded string bytes into utf-8, error: ${e}`)}}function Dx(s){return s.map(e=>Zn(e))}/**
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
 */ex();/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */const Ox=["channelsFirst","channelsLast"],Mx=["nearest","bilinear"],Px=["valid","same","causal"],Rx=["max","avg"];/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */class On extends Error{constructor(e){super(e),Object.setPrototypeOf(this,On.prototype)}}class zn extends Error{constructor(e){super(e),Object.setPrototypeOf(this,zn.prototype)}}class W extends Error{constructor(e){super(e),Object.setPrototypeOf(this,W.prototype)}}class be extends Error{constructor(e){super(e),Object.setPrototypeOf(this,be.prototype)}}class rl extends Error{constructor(e){super(e),Object.setPrototypeOf(this,rl.prototype)}}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function Li(s,e){if(Array.isArray(s)){let t=[];for(let n=0;n<e;n++)t=t.concat(s);return t}else{const t=new Array(e);return t.fill(s),t}}function mn(s,e){if(!s)throw new rl(e)}function Ou(s,e){let t=0;for(const n of s)n===e&&t++;return t}function _t(s){return s.length===1?s[0]:s}function ke(s){return Array.isArray(s)?s:[s]}function In(s){const t=s.replace(/(.)([A-Z][a-z0-9]+)/g,"$1_$2").replace(/([a-z])([A-Z])/g,"$1_$2").toLowerCase();return t[0]!=="_"?t:"private"+t}function Yn(s){return s.length<=1||s.indexOf("_")===-1?s:s.replace(/[_]+(\w|$)/g,(e,t)=>t.toUpperCase())}let Bt={};function il(s){if(s==null)return null;const e={};return e.className=s.getClassName(),e.config=s.getConfig(),e}function _a(s){if(!(s==null||typeof s!="object"))if(Array.isArray(s))s.forEach(e=>_a(e));else{const e=Object.keys(s);for(const t of e){const n=s[t];n!=null&&typeof n=="object"&&(!Array.isArray(n)&&n.type==="ndarray"&&typeof n.value=="number"?s[t]=n.value:_a(n))}}}function Dr(s,e={},t={},n="object",r=!1){if(typeof s=="string"){const i=s;let o;if(i in t)o=t[i];else if(i in Bt)o=Bt[i];else if(o=e[i],o==null)throw new W(`Unknown ${n}: ${s}. This may be due to one of the following reasons:
1. The ${n} is defined in Python, in which case it needs to be ported to TensorFlow.js or your JavaScript code.
2. The custom ${n} is defined in JavaScript, but is not registered properly with tf.serialization.registerClass().`);return o}else{const i=s;if(i.className==null||i.config==null)throw new W(`${n}: Improper config format: ${JSON.stringify(i)}.
'className' and 'config' must set.`);const o=i.className;let a,l;if(o in t?[a,l]=t[o]:o in Bt?[a,l]=Bt.className:o in e&&([a,l]=e[o]),a==null)throw new W(`Unknown ${n}: ${o}. This may be due to one of the following reasons:
1. The ${n} is defined in Python, in which case it needs to be ported to TensorFlow.js or your JavaScript code.
2. The custom ${n} is defined in JavaScript, but is not registered properly with tf.serialization.registerClass().`);if(l!=null){const u={};for(const w of Object.keys(Bt))u[w]=Bt[w];for(const w of Object.keys(t))u[w]=t[w];const c=i.config;c.customObjects=u;const h=Object.assign({},Bt);for(const w of Object.keys(t))Bt[w]=t[w];_a(i.config);const d=l(a,i.config,t,r);return Bt=Object.assign({},h),d}else{const u=Object.assign({},Bt);for(const h of Object.keys(t))Bt[h]=t[h];const c=new a(i.config);return Bt=Object.assign({},u),c}}}function Lx(s,e){return s<e?-1:s>e?1:0}function Yr(s,e){return-1*Lx(s,e)}function ns(s){if(s==null)return s;const e=[];for(const t of s)e.indexOf(t)===-1&&e.push(t);return e}function Bx(s){if(s==null)throw new W(`Invalid value in obj: ${JSON.stringify(s)}`);for(const e in s)if(s.hasOwnProperty(e))return!1;return!0}function Ws(s,e,t){if(t!=null&&s.indexOf(t)<0)throw new W(`${t} is not a valid ${e}.  Valid values are ${s} or null/undefined.`)}function ol(s,e,t=0,n=1/0){return mn(t>=0),mn(n>=t),Array.isArray(s)&&s.length>=t&&s.length<=n&&s.every(r=>typeof r===e)}function $n(s,e){Array.isArray(s)?(P(s.length>0,()=>`${e} is unexpectedly an empty array.`),s.forEach((t,n)=>$n(t,`element ${n+1} of ${e}`))):P(Number.isInteger(s)&&s>0,()=>`Expected ${e} to be a positive integer, but got ${mf(s)}.`)}function mf(s){return s===null?"null":Array.isArray(s)?"["+s.map(e=>mf(e)).join(",")+"]":typeof s=="string"?`"${s}"`:`${s}`}function Fx(s,e,t){let n=t!=null?t():$s(),r;return(...o)=>{const a=t!=null?t():$s();return a-n<e||(n=a,r=s(...o)),r}}function Ux(s){return s==="relu"?"relu":s==="linear"?"linear":s==="elu"?"elu":null}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */const bs=new Map;function Ve(s){Ws(Ox,"DataFormat",s)}function zx(s){Ws(Mx,"InterpolationFormat",s)}function jt(s){Ws(Px,"PaddingMode",s)}function gf(s){Ws(Rx,"PoolMode",s)}const cr=[],Mu="/";function _i(s,e){cr.push(s);try{const t=e();return cr.pop(),t}catch(t){throw cr.pop(),t}}function Vx(){return cr.length===0?"":cr.join(Mu)+Mu}function yf(s){if(!wf(s))throw new Error("Not a valid tensor name: '"+s+"'");return Vx()+s}function bf(s){if(!wf(s))throw new Error("Not a valid tensor name: '"+s+"'");bs.has(s)||bs.set(s,0);const e=bs.get(s);if(bs.set(s,bs.get(s)+1),e>0){const t=`${s}_${e}`;return bs.set(t,1),t}else return s}const Gx=new RegExp(/^[A-Za-z0-9][-A-Za-z0-9\._\/]*$/);function wf(s){return!!s.match(Gx)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function Wx(s){return s===parseInt(s.toString(),10)}function hr(s,e,t){e==null&&(e=0),t==null&&(t=s.length);let n=1;for(let r=e;r<t;++r)n*=s[r];return n}function xf(s){if(s.length===0)return Number.NaN;let e=Number.NEGATIVE_INFINITY;for(let t=0;t<s.length;t++){const n=s[t];n>e&&(e=n)}return e}function Bi(s,e){if(e<s)throw new W(`end (${e}) < begin (${s}) is forbidden.`);const t=[];for(let n=s;n<e;++n)t.push(n);return t}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */let Vo;function ze(){return Vo==null&&(Vo=qy().epsilon()),Vo}function qs(){return"channelsLast"}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function vf(s,e){return Ee(s,e)}function al(s,e=-1){const t=s.shape.slice();return e<0&&(e=t.length+e+1),t.splice(e,0,1),se(s,t)}function qx(s){const e=[hr(s.shape)];return se(s,e)}function ss(s,e,t){return Y(()=>{switch(s.rank){case 1:return el(s,e,t);case 2:return nf(s,[e,0],[t,s.shape[1]]);case 3:return tl(s,[e,0,0],[t,s.shape[1],s.shape[2]]);case 4:return yr(s,[e,0,0,0],[t,s.shape[1],s.shape[2],s.shape[3]]);case 5:return Je(s,[e,0,0,0,0],[t,s.shape[1],s.shape[2],s.shape[3],s.shape[4]]);case 6:return Je(s,[e,0,0,0,0,0],[t,s.shape[1],s.shape[2],s.shape[3],s.shape[4],s.shape[5]]);default:throw new W(`sliceAlongFirstAxis() received an unsupported tensor rank: ${s.rank}`)}})}function Go(s,e,t){return Y(()=>{switch(s.rank){case 1:return el(s,e,t);case 2:return nf(s,[0,e],[s.shape[0],t]);case 3:return tl(s,[0,0,e],[s.shape[0],s.shape[1],t]);case 4:return yr(s,[0,0,0,e],[s.shape[0],s.shape[1],s.shape[2],t]);default:throw new W(`sliceAlongLastAxis() received an unsupported tensor rank: ${s.rank}`)}})}function Qr(s,e,t,n){return Y(()=>{switch(s.rank){case 1:return el(s,e,t);case 2:switch(n){case 1:return ss(s,e,t);case 2:return Go(s,e,t);default:throw new W(`The axis is not within the rank of the tensor ${n}`)}case 3:switch(n){case 1:return ss(s,e,t);case 2:return tl(s,[0,e,0],[s.shape[0],t,s.shape[2]]);case 3:return Go(s,e,t);default:throw new W(`The axis is not within the rank of the tensor ${n}`)}case 4:switch(n){case 1:return ss(s,e,t);case 2:return yr(s,[0,e,0,0],[s.shape[0],t,s.shape[2],s.shape[3]]);case 3:return yr(s,[0,0,e,0],[s.shape[0],s.shape[1],t,s.shape[3]]);case 4:return Go(s,e,t);default:throw new W(`The axis is not within the rank of the tensor ${n}`)}default:throw new W(`sliceAlongLastAxis() received an unsupported tensor rank: ${s.rank}`)}})}function Hx(s,e=-1){let t;return e<0&&(t=s[0].rank,t!==0?e=t:e=0),e===s[0].rank&&(e=-1),es(s,e)}function _f(s,e=0,t=1,n,r){return Ib(s,e,t,n,r)}function jx(s,e,t){return Y(()=>(Array.isArray(e)?e=pt(e,"int32"):e=Ee(e,"int32"),T1(s,e,t)))}function Or(s){return J(s,s)}function Kx(s,e,t){const n=e.shape;if(e.rank!==1&&e.rank!==s)throw new W(`Unexpected bias dimensions: ${e.rank}; expected it to be 1 or ${s}`);if(s===5){if(t==="channelsFirst")return n.length===1?se(e,[1,n[0],1,1,1]):se(e,[1,n[3],n[0],n[1],n[2]]);if(t==="channelsLast")return n.length===1?se(e,[1,1,1,1,n[0]]):se(e,[1].concat(n))}else if(s===4){if(t==="channelsFirst")return n.length===1?se(e,[1,n[0],1,1]):se(e,[1,n[2],n[0],n[1]]);if(t==="channelsLast")return n.length===1?se(e,[1,1,1,n[0]]):se(e,[1].concat(n))}else if(s===3){if(t==="channelsFirst")return n.length===1?se(e,[1,n[0],1]):se(e,[1,n[1],n[0]]);if(t==="channelsLast")return n.length===1?se(e,[1,1,n[0]]):se(e,[1].concat(n))}else if(s<3)return e;throw new W(`Unsupported input rank by biasAdd: ${e.rank}`)}function Mr(s,e,t){return Y(()=>(t==null&&(t=qs()),Ve(t),ae(s,Kx(s.rank,e,t))))}function Xx(s,e=1){if(e!==1)throw new be(`Support for alpha values other than 1 (${e}) is not implemented yet.`);return qh(s)}function Yx(s){return Y(()=>ge(s,ae(dt(s),1)))}function Qx(s){return Y(()=>{const e=ae(.5,J(.2,s));return nn(e,0,1)})}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */class tt extends Gs{getConfig(){return{}}}class Sf extends tt{apply(e,t=1){return Xx(e,t)}}Sf.className="elu";re(Sf);class If extends tt{apply(e){return Mb(e)}}If.className="selu";re(If);class kf extends tt{apply(e){return Nr(e)}}kf.className="relu";re(kf);class Tf extends tt{apply(e){return Y(()=>Mi(6,Nr(e)))}}Tf.className="relu6";re(Tf);class Ef extends tt{apply(e){return e}}Ef.className="linear";re(Ef);class Af extends tt{apply(e){return Ka(e)}}Af.className="sigmoid";re(Af);class Cf extends tt{apply(e){return Qx(e)}}Cf.className="hardSigmoid";re(Cf);class $f extends tt{apply(e){return Za(e)}}$f.className="softplus";re($f);class Nf extends tt{apply(e){return Yx(e)}}Nf.className="softsign";re(Nf);class Df extends tt{apply(e){return Xa(e)}}Df.className="tanh";re(Df);class Of extends tt{apply(e,t=-1){return sf(e,t)}}Of.className="softmax";re(Of);class Mf extends tt{apply(e,t=-1){return W1(e,t)}}Mf.className="logSoftmax";re(Mf);class Pf extends tt{apply(e){return Y(()=>Y(()=>{const t=Math.sqrt(2),n=J(.5,ae(1,a1(ge(e,t))));return J(e,n)}))}}Pf.className="gelu";re(Pf);class Rf extends tt{apply(e){return Y(()=>J(.5,J(e,ae(1,Xa(J(sn(ge(2,Math.PI)),ae(e,J(.044715,Oi(e,3)))))))))}}Rf.className="gelu_new";re(Rf);class Lf extends tt{apply(e){return Y(()=>J(e,Xa(Za(e))))}}Lf.className="mish";re(Lf);class Bf extends tt{apply(e,t=1){return Y(()=>J(Ka(J(e,t)),e))}}Bf.className="swish";re(Bf);function Zx(s){return s.getClassName()}function Wo(s,e={}){return Dr(s,Vt.getMap().classNameMap,e,"activation")}function Jx(s){if(s==null){const e={};return e.className="linear",e.config={},Wo(e)}if(typeof s=="string"){const e={};return e.className=s,e.config={},Wo(e)}else return s instanceof tt?s:Wo(s)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function ll(s,e){return Y(()=>sn(Se(J(s,s),e,!0)))}class Pr extends Gs{getConfig(){return{}}}class Ff extends Pr{constructor(e){super(),this.defaultMaxValue=2,this.defaultAxis=0,this.maxValue=e.maxValue!=null?e.maxValue:this.defaultMaxValue,this.axis=e.axis!=null?e.axis:this.defaultAxis}apply(e){return Y(()=>{const t=ll(e,this.axis),n=nn(t,0,this.maxValue);return J(e,ge(n,ae(ze(),t)))})}getConfig(){return{maxValue:this.maxValue,axis:this.axis}}}Ff.className="MaxNorm";re(Ff);class Uf extends Pr{constructor(e){super(),this.defaultAxis=0,this.axis=e.axis!=null?e.axis:this.defaultAxis}apply(e){return Y(()=>ge(e,ae(ze(),ll(e,this.axis))))}getConfig(){return{axis:this.axis}}}Uf.className="UnitNorm";re(Uf);class zf extends Pr{apply(e){return Nr(e)}}zf.className="NonNeg";re(zf);class Vf extends Pr{constructor(e){super(),this.defaultMinValue=0,this.defaultMaxValue=1,this.defaultRate=1,this.defaultAxis=0,this.minValue=e.minValue!=null?e.minValue:this.defaultMinValue,this.maxValue=e.maxValue!=null?e.maxValue:this.defaultMaxValue,this.rate=e.rate!=null?e.rate:this.defaultRate,this.axis=e.axis!=null?e.axis:this.defaultAxis}apply(e){return Y(()=>{const t=ll(e,this.axis),n=ae(J(this.rate,nn(t,this.minValue,this.maxValue)),J(1-this.rate,t));return J(e,ge(n,ae(ze(),t)))})}getConfig(){return{minValue:this.minValue,maxValue:this.maxValue,rate:this.rate,axis:this.axis}}}Vf.className="MinMaxNorm";re(Vf);const Pu={maxNorm:"MaxNorm",minMaxNorm:"MinMaxNorm",nonNeg:"NonNeg",unitNorm:"UnitNorm"};function Fi(s){return il(s)}function Ru(s,e={}){return Dr(s,Vt.getMap().classNameMap,e,"constraint")}function Ui(s){if(s==null)return null;if(typeof s=="string"){const t={className:s in Pu?Pu[s]:s,config:{}};return Ru(t)}else return s instanceof Pr?s:Ru(s)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */let e2=0;function Gf(){return e2++}const Zr={};function ul(s=""){return s in Zr||(Zr[s]=0),Zr[s]+=1,s+Zr[s].toString()}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */const t2=["fanIn","fanOut","fanAvg"],n2=["normal","uniform","truncatedNormal"];/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function s2(s){Ws(t2,"FanMode",s)}function r2(s){Ws(n2,"Distribution",s)}class _n extends Gs{fromConfigUsesCustomObjects(){return!1}getConfig(){return{}}}class Wf extends _n{apply(e,t){return Os(e,t)}}Wf.className="Zeros";re(Wf);class qf extends _n{apply(e,t){return Ja(e,t)}}qf.className="Ones";re(qf);class Hf extends _n{constructor(e){if(super(),typeof e!="object")throw new W(`Expected argument of type ConstantConfig but got ${e}`);if(e.value===void 0)throw new W(`config must have value set but got ${e}`);this.value=e.value}apply(e,t){return Y(()=>J(Rt(this.value),Ja(e,t)))}getConfig(){return{value:this.value}}}Hf.className="Constant";re(Hf);class jf extends _n{constructor(e){super(),this.DEFAULT_MINVAL=-.05,this.DEFAULT_MAXVAL=.05,this.minval=e.minval||this.DEFAULT_MINVAL,this.maxval=e.maxval||this.DEFAULT_MAXVAL,this.seed=e.seed}apply(e,t){return tf(e,this.minval,this.maxval,t,this.seed)}getConfig(){return{minval:this.minval,maxval:this.maxval,seed:this.seed}}}jf.className="RandomUniform";re(jf);class Kf extends _n{constructor(e){super(),this.DEFAULT_MEAN=0,this.DEFAULT_STDDEV=.05,this.mean=e.mean||this.DEFAULT_MEAN,this.stddev=e.stddev||this.DEFAULT_STDDEV,this.seed=e.seed}apply(e,t){if(t=t||"float32",t!=="float32"&&t!=="int32")throw new be(`randomNormal does not support dType ${t}.`);return _f(e,this.mean,this.stddev,t,this.seed)}getConfig(){return{mean:this.mean,stddev:this.stddev,seed:this.seed}}}Kf.className="RandomNormal";re(Kf);class Xf extends _n{constructor(e){super(),this.DEFAULT_MEAN=0,this.DEFAULT_STDDEV=.05,this.mean=e.mean||this.DEFAULT_MEAN,this.stddev=e.stddev||this.DEFAULT_STDDEV,this.seed=e.seed}apply(e,t){if(t=t||"float32",t!=="float32"&&t!=="int32")throw new be(`truncatedNormal does not support dType ${t}.`);return of(e,this.mean,this.stddev,t,this.seed)}getConfig(){return{mean:this.mean,stddev:this.stddev,seed:this.seed}}}Xf.className="TruncatedNormal";re(Xf);class Yf extends _n{constructor(e){super(),this.gain=e.gain!=null?e.gain:1}apply(e,t){return Y(()=>{if(e.length!==2||e[0]!==e[1])throw new W("Identity matrix initializer can only be used for 2D square matrices.");return J(this.gain,Yh(e[0]))})}getConfig(){return{gain:this.gain}}}Yf.className="Identity";re(Yf);function i2(s,e="channelsLast"){let t,n;if(Ve(e),s.length===2)t=s[0],n=s[1];else if([3,4,5].indexOf(s.length)!==-1){if(e==="channelsFirst"){const r=hr(s,2);t=s[1]*r,n=s[0]*r}else if(e==="channelsLast"){const r=hr(s,0,s.length-2);t=s[s.length-2]*r,n=s[s.length-1]*r}}else{const r=hr(s);t=Math.sqrt(r),n=Math.sqrt(r)}return[t,n]}class It extends _n{constructor(e){if(super(),e.scale<0)throw new W(`scale must be a positive float. Got: ${e.scale}`);this.scale=e.scale==null?1:e.scale,this.mode=e.mode==null?"fanIn":e.mode,s2(this.mode),this.distribution=e.distribution==null?"normal":e.distribution,r2(this.distribution),this.seed=e.seed}apply(e,t){const n=i2(e),r=n[0],i=n[1];let o=this.scale;if(this.mode==="fanIn"?o/=Math.max(1,r):this.mode==="fanOut"?o/=Math.max(1,i):o/=Math.max(1,(r+i)/2),this.distribution==="normal"){const a=Math.sqrt(o);if(t=t||"float32",t!=="float32"&&t!=="int32")throw new be(`${this.getClassName()} does not support dType ${t}.`);return of(e,0,a,t,this.seed)}else{const a=Math.sqrt(3*o);return tf(e,-a,a,t,this.seed)}}getConfig(){return{scale:this.scale,mode:this.mode,distribution:this.distribution,seed:this.seed}}}It.className="VarianceScaling";re(It);class cl extends It{constructor(e){super({scale:1,mode:"fanAvg",distribution:"uniform",seed:e==null?null:e.seed})}getClassName(){return It.className}}cl.className="GlorotUniform";re(cl);class hl extends It{constructor(e){super({scale:1,mode:"fanAvg",distribution:"normal",seed:e==null?null:e.seed})}getClassName(){return It.className}}hl.className="GlorotNormal";re(hl);class fl extends It{constructor(e){super({scale:2,mode:"fanIn",distribution:"normal",seed:e==null?null:e.seed})}getClassName(){return It.className}}fl.className="HeNormal";re(fl);class dl extends It{constructor(e){super({scale:2,mode:"fanIn",distribution:"uniform",seed:e==null?null:e.seed})}getClassName(){return It.className}}dl.className="HeUniform";re(dl);class pl extends It{constructor(e){super({scale:1,mode:"fanIn",distribution:"normal",seed:e==null?null:e.seed})}getClassName(){return It.className}}pl.className="LeCunNormal";re(pl);class ml extends It{constructor(e){super({scale:1,mode:"fanIn",distribution:"uniform",seed:e==null?null:e.seed})}getClassName(){return It.className}}ml.className="LeCunUniform";re(ml);class Qf extends _n{constructor(e){super(),this.DEFAULT_GAIN=1,this.ELEMENTS_WARN_SLOW=2e3,this.gain=e.gain==null?this.DEFAULT_GAIN:e.gain,this.seed=e.seed}apply(e,t){return Y(()=>{if(e.length<2)throw new be("Shape must be at least 2D.");if(t!=="int32"&&t!=="float32"&&t!==void 0)throw new TypeError(`Unsupported data type ${t}.`);t=t;const n=he(e.slice(0,-1)),r=e[e.length-1],i=n*r;i>this.ELEMENTS_WARN_SLOW&&console.warn(`Orthogonal initializer is being called on a matrix with more than ${this.ELEMENTS_WARN_SLOW} (${i}) elements: Slowness may result.`);const o=[Math.max(r,n),Math.min(r,n)],a=_f(o,0,1,t,this.seed),l=Yw.qr(a,!1);let u=l[0];const h=l[1].flatten().stridedSlice([0],[Math.min(r,n)*Math.min(r,n)],[Math.min(r,n)+1]);return u=J(u,h.sign()),n<r&&(u=u.transpose()),J(Rt(this.gain),u.reshape(e))})}getConfig(){return{gain:this.gain,seed:this.seed}}}Qf.className="Orthogonal";re(Qf);const Lu={constant:"Constant",glorotNormal:"GlorotNormal",glorotUniform:"GlorotUniform",heNormal:"HeNormal",heUniform:"HeUniform",identity:"Identity",leCunNormal:"LeCunNormal",leCunUniform:"LeCunUniform",ones:"Ones",orthogonal:"Orthogonal",randomNormal:"RandomNormal",randomUniform:"RandomUniform",truncatedNormal:"TruncatedNormal",varianceScaling:"VarianceScaling",zeros:"Zeros"};function Bu(s,e={}){return Dr(s,Vt.getMap().classNameMap,e,"initializer")}function zi(s){return il(s)}function xr(s){if(typeof s=="string"){const e=s in Lu?Lu[s]:s;if(e==="GlorotNormal")return new hl;if(e==="GlorotUniform")return new cl;if(e==="HeNormal")return new fl;if(e==="HeUniform")return new dl;if(e==="LeCunNormal")return new pl;if(e==="LeCunUniform")return new ml;{const t={};return t.className=e,t.config={},Bu(t)}}else return s instanceof _n?s:Bu(s)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function Vi(s){return s.length===0?[]:Array.isArray(s[0])?s:[s]}function kt(s){let e;if(Array.isArray(s)){if(s.length!==1)throw new W(`Expected Tensor length to be 1; got ${s.length}`);e=s[0]}else e=s;return e}function rn(s){if(Array.isArray(s)&&Array.isArray(s[0])){if(s.length===1)return s=s,s[0];throw new W(`Expected exactly 1 Shape; got ${s.length}`)}else return s}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function Gi(s){let e=0;for(const t of s)t.shape.length===0?e+=1:e+=t.shape.reduce((n,r)=>n*r);return e}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */const Fu="Variable";class o2{constructor(e,t="float32",n=Fu,r=!0,i=null){this.dtype=t??"float32",this.shape=e.shape,this.id=Gf(),n=n??Fu,this.originalName=yf(n),this.name=bf(this.originalName),this.trainable_=r,this.constraint=i,this.val=Kb(e,this.trainable_,this.name,this.dtype)}read(){return this.assertNotDisposed(),this.val}write(e){return this.assertNotDisposed(),a2(this.val,e),this.val.id!==e.id&&(this.val.assign(e),this.constraint!=null&&this.val.assign(this.constraint.apply(this.val))),this}dispose(){this.assertNotDisposed(),this.val.dispose()}assertNotDisposed(){if(this.val.isDisposed)throw new Error(`LayersVariable ${this.name} is already disposed.`)}get trainable(){return this.trainable_}set trainable(e){this.trainable_=e,this.val.trainable=e}}function a2(s,e){if(s.shape.toString()!==e.shape.toString())throw new Error("Shape mismatch: "+JSON.stringify(s.shape)+" vs. "+JSON.stringify(e.shape))}function Uu(s){return s.map(e=>e.read())}function Zf(s){s.forEach(e=>{e[0].write(e[1])})}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */class wn{constructor(e){this.dtype=e.dtype,this.shape=e.shape,e.shape!=null?this.ndim=e.shape.length:this.ndim=e.ndim,this.maxNDim=e.maxNDim,this.minNDim=e.minNDim,this.axes=e.axes||{}}}class fs{constructor(e,t,n,r,i,o,a){this.dtype=e,this.shape=t,this.sourceLayer=n,this.inputs=r,this.callArgs=i,this.outputTensorIndex=a,this.id=Gf(),o!=null&&(this.originalName=yf(o),this.name=bf(this.originalName)),this.rank=t.length}}let l2=0;class gl{constructor(e,t){this.callArgs=t,this.id=l2++,this.outboundLayer=e.outboundLayer,this.inboundLayers=e.inboundLayers,this.nodeIndices=e.nodeIndices,this.tensorIndices=e.tensorIndices,this.inputTensors=e.inputTensors,this.outputTensors=e.outputTensors,this.inputMasks=e.inputMasks,this.outputMasks=e.outputMasks,this.inputShapes=e.inputShapes,this.outputShapes=e.outputShapes;for(const n of e.inboundLayers)n?.outboundNodes.push(this);e.outboundLayer.inboundNodes.push(this)}getConfig(){const e=[];for(const t of this.inboundLayers)t!=null?e.push(t.name):e.push(null);return{outboundLayer:this.outboundLayer?this.outboundLayer.name:null,inboundLayers:e,nodeIndices:this.nodeIndices,tensorIndices:this.tensorIndices}}}let u2=0;class on extends Gs{constructor(e={}){super(),this._callHook=null,this._addedWeightNames=[],this._stateful=!1,this.id=u2++,this.activityRegularizer=null,this.inputSpec=null,this.supportsMasking=!1,this._trainableWeights=[],this._nonTrainableWeights=[],this._losses=[],this._updates=[],this._built=!1,this.inboundNodes=[],this.outboundNodes=[];let t=e.name;if(!t){const n=this.getClassName();t=In(n)+"_"+ul(n)}if(this.name=t,this.trainable_=e.trainable==null?!0:e.trainable,e.inputShape!=null||e.batchInputShape!=null){let n;if(e.batchInputShape!=null)n=e.batchInputShape;else if(e.inputShape!=null){let i=null;e.batchSize!=null&&(i=e.batchSize),n=[i].concat(e.inputShape)}this.batchInputShape=n;let r=e.dtype;r==null&&(r=e.inputDType),r==null&&(r="float32"),this.dtype=r}e.weights!=null?this.initialWeights=e.weights:this.initialWeights=null,this._refCount=null,this.fastWeightInitDuringBuild=!1}static nodeKey(e,t){return e.name+"_ib-"+t.toString()}getNodeAtIndex(e,t){if(this.inboundNodes.length===0)throw new zn(`The layer has never been called and thus has no defined ${t}.`);if(this.inboundNodes.length<=e)throw new W(`Asked to get ${t} at node ${e}, but the layer has only ${this.inboundNodes.length} inbound nodes.`);return this.inboundNodes[e]}getInputAt(e){return _t(this.getNodeAtIndex(e,"input").inputTensors)}getOutputAt(e){return _t(this.getNodeAtIndex(e,"output").outputTensors)}get input(){if(this.inboundNodes.length>1)throw new On(`Layer ${this.name} has multiple inbound nodes, hence the notion of "layer input" is ill-defined. Use \`getInputAt(nodeIndex)\` instead.`);if(this.inboundNodes.length===0)throw new On(`Layer ${this.name} is not connected, no input to return.`);return _t(this.getNodeAtIndex(0,"input").inputTensors)}get output(){if(this.inboundNodes.length===0)throw new On(`Layer ${this.name} has no inbound nodes.`);if(this.inboundNodes.length>1)throw new On(`Layer ${this.name} has multiple inbound nodes, hence the notion of "layer output" is ill-defined. Use \`getOutputAt(nodeIndex)\` instead.`);return _t(this.getNodeAtIndex(0,"output").outputTensors)}get losses(){return this._losses}calculateLosses(){return this.losses.map(e=>e())}get updates(){return this._updates}get built(){return this._built}set built(e){this._built=e}get trainable(){return this.trainable_}set trainable(e){this._trainableWeights.forEach(t=>t.trainable=e),this.trainable_=e}get trainableWeights(){return this.trainable_?this._trainableWeights.filter(e=>e.trainable):[]}set trainableWeights(e){this._trainableWeights=e}get nonTrainableWeights(){return this.trainable?this._trainableWeights.filter(e=>!e.trainable).concat(this._nonTrainableWeights):this._trainableWeights.concat(this._nonTrainableWeights)}set nonTrainableWeights(e){this._nonTrainableWeights=e}get weights(){return this.trainableWeights.concat(this.nonTrainableWeights)}get stateful(){return this._stateful}resetStates(){if(!this.stateful)throw new Error("Cannot call the resetStates() method of a non-stateful Layer object.")}assertInputCompatibility(e){const t=ke(e);if(this.inputSpec==null||this.inputSpec.length===0)return;const n=ke(this.inputSpec);if(t.length!==n.length)throw new W(`Layer ${this.name} expects ${n.length} inputs, but it received ${t.length} input tensors. Input received: ${e}`);for(let r=0;r<t.length;r++){const i=t[r],o=n[r];if(o==null)continue;const a=i.rank;if(o.ndim!=null&&a!==o.ndim)throw new W(`Input ${r} is incompatible with layer ${this.name}: expected ndim=${o.ndim}, found ndim=${a}`);if(o.maxNDim!=null&&a>o.maxNDim)throw new W(`Input ${r} is incompatible with layer ${this.name}: expected max_ndim=${o.maxNDim}, found ndim=${a}`);if(o.minNDim!=null&&a<o.minNDim)throw new W(`Input ${r} is incompatible with layer ${this.name}: expected min_ndim=${o.minNDim}, found ndim=${a}.`);if(o.dtype!=null&&i.dtype!==o.dtype)throw new W(`Input ${r} is incompatible with layer ${this.name} : expected dtype=${o.dtype}, found dtype=${i.dtype}.`);if(o.axes){const l=i.shape;for(const u in o.axes){const c=Number(u),h=o.axes[u],d=c>=0?l[c]:l[l.length+c];if(h!=null&&[h,null].indexOf(d)===-1)throw new W(`Input ${r} is incompatible with layer ${this.name}: expected axis ${c} of input shape to have value ${h} but got shape ${l}.`)}}if(o.shape!=null)for(let l=0;l<o.shape.length;++l){const u=o.shape[l],c=i.shape[l];if(u!=null&&c!=null&&u!==c)throw new W(`Input ${r} is incompatible with layer ${this.name}: expected shape=${o.shape}, found shape=${i.shape}.`)}}}call(e,t){return e}invokeCallHook(e,t){this._callHook!=null&&this._callHook(e,t)}setCallHook(e){this._callHook=e}clearCallHook(){this._callHook=null}apply(e,t){t=t||{},this.assertNotDisposed();const n=ke(e),r=f2(e),i=d2(e);if(r===i)throw new W("Arguments to apply() must be all SymbolicTensors or all Tensors");return _i(this.name,()=>{if(!this.built){this.assertInputCompatibility(e);const o=[];for(const a of ke(e))o.push(a.shape);this.build(_t(o)),this.built=!0,this.initialWeights&&this.setWeights(this.initialWeights),this._refCount===null&&i&&(this._refCount=1)}if(this.assertInputCompatibility(e),i){let o=this.call(e,t);this.supportsMasking&&this.setMaskMetadata(e,o);const a=ke(o),l=[];for(let u of a)n.indexOf(u)!==-1&&(u=u.clone()),l.push(u);if(o=_t(l),this.activityRegularizer!=null)throw new be("Layer invocation in the presence of activity regularizer(s) is not supported yet.");return o}else{const o=c2(e),a=this.computeOutputShape(o);let l;const u=h2(e);if(this.warnOnIncompatibleInputShape(Array.isArray(e)?o[0]:o),a!=null&&a.length>0&&Array.isArray(a[0])?l=a.map((c,h)=>new fs(u,c,this,ke(e),t,this.name,h)):l=new fs(u,a,this,ke(e),t,this.name),this.addInboundNode(e,l,null,null,o,a,t),this._refCount++,this.activityRegularizer!=null)throw new be("Layer invocation in the presence of activity regularizer(s) is not supported yet.");return l}})}warnOnIncompatibleInputShape(e){if(this.batchInputShape!=null)if(e.length!==this.batchInputShape.length)console.warn(`The rank of the input tensor provided (shape: ${JSON.stringify(e)}) does not match that of the batchInputShape (${JSON.stringify(this.batchInputShape)}) of the layer ${this.name}`);else{let t=!1;this.batchInputShape.forEach((n,r)=>{n!=null&&e[r]!=null&&e[r]!==n&&(t=!0)}),t&&console.warn(`The shape of the input tensor (${JSON.stringify(e)}) does not match the expectation of layer ${this.name}: ${JSON.stringify(this.batchInputShape)}`)}}get outputShape(){if(this.inboundNodes==null||this.inboundNodes.length===0)throw new On(`The layer ${this.name} has never been called and thus has no defined output shape.`);const e=[];for(const t of this.inboundNodes){const n=JSON.stringify(t.outputShapes);e.indexOf(n)===-1&&e.push(n)}if(e.length===1){const t=this.inboundNodes[0].outputShapes;return Array.isArray(t)&&Array.isArray(t[0])&&t.length===1?t[0]:t}else throw new On(`The layer ${this.name} has multiple inbound nodes with different output shapes. Hence the notion of "output shape" is ill-defined for the layer.`)}countParams(){if(!this.built)throw new zn(`You tried to call countParams() on ${this.name}, but the layer is not built yet. Build it first by calling build(batchInputShape).`);return Gi(this.weights)}build(e){this.built=!0}getWeights(e=!1){return Uu(e?this.trainableWeights:this.weights)}setWeights(e){Y(()=>{const t=this.weights;if(t.length!==e.length)throw new W(`You called setWeights(weights) on layer "${this.name}" with a weight list of length ${e.length}, but the layer was expecting ${t.length} weights. Provided weights: ${e}...`);if(t.length===0)return;const n=[],r=Uu(t);for(let i=0;i<r.length;++i){const o=r[i],a=t[i],l=e[i];if(!Ht(o.shape,l.shape))throw new W(`Layer weight shape ${o.shape} not compatible with provided weight shape ${l.shape}`);n.push([a,l])}Zf(n)})}addWeight(e,t,n,r,i,o,a,l){if(this._addedWeightNames.indexOf(e)!==-1)throw new W(`Duplicate weight name ${e} for layer ${this.name}`);this._addedWeightNames.push(e),n==null&&(n="float32"),this.fastWeightInitDuringBuild&&(r=l!=null?l():xr("zeros"));const u=r.apply(t,n),c=new o2(u,n,e,o,a);return u.dispose(),i!=null&&this.addLoss(()=>i.apply(c.read())),o==null&&(o=!0),o?this._trainableWeights.push(c):this._nonTrainableWeights.push(c),c}setFastWeightInitDuringBuild(e){this.fastWeightInitDuringBuild=e}addLoss(e){e==null||Array.isArray(e)&&e.length===0||(e=ke(e),this._losses!==void 0&&this._losses!==null&&this.losses.push(...e))}computeOutputShape(e){return e}computeMask(e,t){if(!this.supportsMasking){if(t!=null)if(Array.isArray(t))t.forEach(n=>{if(n!=null)throw new TypeError(`Layer ${this.name} does not support masking, but was passed an inputMask.`)});else throw new TypeError(`Layer ${this.name} does not support masking, but was passed an inputMask.`);return null}return t}setMaskMetadata(e,t,n){if(!this.supportsMasking)return;const r=this.computeMask(e,n),i=ke(t),o=ke(r);if(i.length!==o.length)throw new Error(`${this.name} outputs ${i.length} tensors but ${i.length} masks for those tensors`);for(let a=0;a<i.length;a++)i[a].kerasMask=o[a]}addInboundNode(e,t,n,r,i,o,a=null){const l=ke(e);t=ke(t),n=ke(n),r=ke(r),i=Vi(i),o=Vi(o);const u=[],c=[],h=[];for(const d of l)u.push(d.sourceLayer),c.push(d.nodeIndex),h.push(d.tensorIndex);new gl({outboundLayer:this,inboundLayers:u,nodeIndices:c,tensorIndices:h,inputTensors:l,outputTensors:t,inputMasks:n,outputMasks:r,inputShapes:i,outputShapes:o},a);for(let d=0;d<t.length;d++)t[d].sourceLayer=this,t[d].nodeIndex=this.inboundNodes.length-1,t[d].tensorIndex=d}getConfig(){const e={name:this.name,trainable:this.trainable};return this.batchInputShape!=null&&(e.batchInputShape=this.batchInputShape),this.dtype!=null&&(e.dtype=this.dtype),e}disposeWeights(){return this.weights.forEach(e=>e.dispose()),this.weights.length}assertNotDisposed(){if(this._refCount===0)throw new Error(`Layer '${this.name}' is already disposed.`)}dispose(){if(!this.built)throw new Error(`Cannot dispose Layer ${this.name} because it has not been built yet.`);if(this._refCount===null)throw new Error(`Cannot dispose Layer ${this.name} because it has not been used yet.`);this.assertNotDisposed();let e=0;return--this._refCount===0&&(e=this.disposeWeights()),{refCountAfterDispose:this._refCount,numDisposedVariables:e}}}function c2(s){s=ke(s);const e=[];for(const t of s)e.push(t.shape);return _t(e)}function h2(s){return"float32"}function f2(s){let e=!0;for(const t of ke(s))if(!(t instanceof fs)){e=!1;break}return e}function d2(s){let e=!0;for(const t of ke(s))if(t instanceof fs){e=!1;break}return e}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function p2(s){if(s!=null&&typeof s!="object")throw new Error(`Argument to L1L2 regularizer's constructor is expected to be an object, but received: ${s}`)}class Jf extends Gs{}class ed extends Jf{constructor(e){super(),p2(e),this.l1=e==null||e.l1==null?.01:e.l1,this.l2=e==null||e.l2==null?.01:e.l2,this.hasL1=this.l1!==0,this.hasL2=this.l2!==0}apply(e){return Y(()=>{let t=Os([1]);return this.hasL1&&(t=ae(t,Se(J(this.l1,dt(e))))),this.hasL2&&(t=ae(t,Se(J(this.l2,Or(e))))),se(t,[])})}getConfig(){return{l1:this.l1,l2:this.l2}}static fromConfig(e,t){return new e({l1:t.l1,l2:t.l2})}}ed.className="L1L2";re(ed);const zu={l1l2:"L1L2"};function vr(s){return il(s)}function Vu(s,e={}){return Dr(s,Vt.getMap().classNameMap,e,"regularizer")}function _r(s){if(s==null)return null;if(typeof s=="string"){const t={className:s in zu?zu[s]:s,config:{}};return Vu(t)}else return s instanceof Jf?s:Vu(s)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function qo(s,e,t){if(typeof s=="number")return Li(s,e);if(s.length!==e)throw new W(`The ${t} argument must be an integer or tuple of ${e} integers. Received: ${s.length} elements.`);for(let n=0;n<e;++n){const r=s[n];if(!Wx(r))throw new W(`The ${t} argument must be an integer or tuple of ${e} integers. Received: ${JSON.stringify(s)} including a non-integer number ${r}`)}return s}function rs(s,e,t,n,r=1){if(s==null)return s;const i=e+(e-1)*(r-1);let o;return t==="same"?o=s:o=s-i+1,Math.floor((o+n-1)/n)}function gn(s,e,t,n){if(s==null)return null;if(n==="valid")s=s*e+xf([t-e,0]);else if(n==="same")s=s*e;else throw new W(`Unsupport padding mode: ${n}.`);return s}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function td(s,e){return Y(()=>(Ve(e),e==="channelsFirst"?Be(s,[0,2,3,1]):s))}function nd(s,e){return Y(()=>(Ve(e),e==="channelsFirst"?Be(s,[0,2,3,4,1]):s))}function m2(s,e,t,n=1,r="valid",i,o=1){return Y(()=>{if(i==null&&(i=qs()),Ve(i),s.shape.length!==3)throw new W(`The input of a conv1dWithBias operation should be 3, but is ${s.shape.length} instead.`);if(e.shape.length!==3)throw new W(`The kernel for a conv1dWithBias operation should be 3, but is ${e.shape.length} instead`);if(t!=null&&t.shape.length!==1)throw new W(`The bias for a conv1dWithBias operation should be 1, but is ${t.shape.length} instead`);if(i==="channelsFirst"&&(s=Be(s,[0,2,1])),r==="causal")throw new be("The support for CAUSAL padding mode in conv1dWithBias is not implemented yet.");let a=V0(s,e,n,r==="same"?"same":"valid","NWC",o);return t!=null&&(a=Mr(a,t)),a})}function Gu(s,e,t,n=[1,1],r="valid",i,o,a=null){return Y(()=>{if(i==null&&(i=qs()),Ve(i),s.rank!==3&&s.rank!==4)throw new W(`conv2dWithBiasActivation expects input to be of rank 3 or 4, but received ${s.rank}.`);if(e.rank!==3&&e.rank!==4)throw new W(`conv2dWithBiasActivation expects kernel to be of rank 3 or 4, but received ${s.rank}.`);let l=td(s,i);if(r==="causal")throw new be("The support for CAUSAL padding mode in conv1dWithBias is not implemented yet.");return l=sw({x:l,filter:e,strides:n,pad:r==="same"?"same":"valid",dilations:o,dataFormat:"NHWC",bias:t,activation:a}),i==="channelsFirst"&&(l=Be(l,[0,3,1,2])),l})}function g2(s,e,t,n=[1,1,1],r="valid",i,o){return Y(()=>{if(i==null&&(i=qs()),Ve(i),s.rank!==4&&s.rank!==5)throw new W(`conv3dWithBias expects input to be of rank 4 or 5, but received ${s.rank}.`);if(e.rank!==4&&e.rank!==5)throw new W(`conv3dWithBias expects kernel to be of rank 4 or 5, but received ${s.rank}.`);let a=nd(s,i);if(r==="causal")throw new be("The support for CAUSAL padding mode in conv3dWithBias is not implemented yet.");return a=j0(a,e,n,r==="same"?"same":"valid","NDHWC",o),t!=null&&(a=Mr(a,t)),i==="channelsFirst"&&(a=Be(a,[0,4,1,2,3])),a})}class yl extends on{constructor(e,t){if(super(t),this.bias=null,this.DEFAULT_KERNEL_INITIALIZER="glorotNormal",this.DEFAULT_BIAS_INITIALIZER="zeros",yl.verifyArgs(t),this.rank=e,$n(this.rank,"rank"),this.rank!==1&&this.rank!==2&&this.rank!==3)throw new be(`Convolution layer for rank other than 1, 2, or 3 (${this.rank}) is not implemented yet.`);if(this.kernelSize=qo(t.kernelSize,e,"kernelSize"),this.strides=qo(t.strides==null?1:t.strides,e,"strides"),this.padding=t.padding==null?"valid":t.padding,jt(this.padding),this.dataFormat=t.dataFormat==null?"channelsLast":t.dataFormat,Ve(this.dataFormat),this.activation=Jx(t.activation),this.useBias=t.useBias==null?!0:t.useBias,this.biasInitializer=xr(t.biasInitializer||this.DEFAULT_BIAS_INITIALIZER),this.biasConstraint=Ui(t.biasConstraint),this.biasRegularizer=_r(t.biasRegularizer),this.activityRegularizer=_r(t.activityRegularizer),this.dilationRate=qo(t.dilationRate==null?1:t.dilationRate,e,"dilationRate"),this.rank===1&&Array.isArray(this.dilationRate)&&this.dilationRate.length!==1)throw new W(`dilationRate must be a number or an array of a single number for 1D convolution, but received ${JSON.stringify(this.dilationRate)}`);if(this.rank===2){if(typeof this.dilationRate=="number")this.dilationRate=[this.dilationRate,this.dilationRate];else if(this.dilationRate.length!==2)throw new W(`dilationRate must be a number or array of two numbers for 2D convolution, but received ${JSON.stringify(this.dilationRate)}`)}else if(this.rank===3){if(typeof this.dilationRate=="number")this.dilationRate=[this.dilationRate,this.dilationRate,this.dilationRate];else if(this.dilationRate.length!==3)throw new W(`dilationRate must be a number or array of three numbers for 3D convolution, but received ${JSON.stringify(this.dilationRate)}`)}}static verifyArgs(e){if(mn("kernelSize"in e,"required key 'kernelSize' not in config"),typeof e.kernelSize!="number"&&!ol(e.kernelSize,"number",1,3))throw new W(`BaseConv expects config.kernelSize to be number or number[] with length 1, 2, or 3, but received ${JSON.stringify(e.kernelSize)}.`)}getConfig(){const e={kernelSize:this.kernelSize,strides:this.strides,padding:this.padding,dataFormat:this.dataFormat,dilationRate:this.dilationRate,activation:Zx(this.activation),useBias:this.useBias,biasInitializer:zi(this.biasInitializer),biasRegularizer:vr(this.biasRegularizer),activityRegularizer:vr(this.activityRegularizer),biasConstraint:Fi(this.biasConstraint)},t=super.getConfig();return Object.assign(e,t),e}}class Hs extends yl{constructor(e,t){super(e,t),this.kernel=null,Hs.verifyArgs(t),this.filters=t.filters,$n(this.filters,"filters"),this.kernelInitializer=xr(t.kernelInitializer||this.DEFAULT_KERNEL_INITIALIZER),this.kernelConstraint=Ui(t.kernelConstraint),this.kernelRegularizer=_r(t.kernelRegularizer)}build(e){e=rn(e);const t=this.dataFormat==="channelsFirst"?1:e.length-1;if(e[t]==null)throw new W(`The channel dimension of the input should be defined. Found ${e[t]}`);const n=e[t],r=this.kernelSize.concat([n,this.filters]);this.kernel=this.addWeight("kernel",r,null,this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight("bias",[this.filters],null,this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint)),this.inputSpec=[{ndim:this.rank+2,axes:{[t]:n}}],this.built=!0}call(e,t){return Y(()=>{e=kt(e);let n;const r=this.bias==null?null:this.bias.read(),i=Ux(this.activation.getClassName());if(i!=null&&this.rank===2)n=Gu(e,this.kernel.read(),r,this.strides,this.padding,this.dataFormat,this.dilationRate,i);else{if(this.rank===1)n=m2(e,this.kernel.read(),r,this.strides[0],this.padding,this.dataFormat,this.dilationRate[0]);else if(this.rank===2)n=Gu(e,this.kernel.read(),r,this.strides,this.padding,this.dataFormat,this.dilationRate);else if(this.rank===3)n=g2(e,this.kernel.read(),r,this.strides,this.padding,this.dataFormat,this.dilationRate);else throw new be("convolutions greater than 3D are not implemented yet.");this.activation!=null&&(n=this.activation.apply(n))}return n})}computeOutputShape(e){e=rn(e);const t=[],n=this.dataFormat==="channelsLast"?e.slice(1,e.length-1):e.slice(2);for(let i=0;i<n.length;++i){const o=rs(n[i],this.kernelSize[i],this.padding,this.strides[i],typeof this.dilationRate=="number"?this.dilationRate:this.dilationRate[i]);t.push(o)}let r=[e[0]];return this.dataFormat==="channelsLast"?(r=r.concat(t),r.push(this.filters)):(r.push(this.filters),r=r.concat(t)),r}getConfig(){const e={filters:this.filters,kernelInitializer:zi(this.kernelInitializer),kernelRegularizer:vr(this.kernelRegularizer),kernelConstraint:Fi(this.kernelConstraint)},t=super.getConfig();return Object.assign(e,t),e}static verifyArgs(e){if(!("filters"in e)||typeof e.filters!="number"||e.filters<1)throw new W(`Convolution layer expected config.filters to be a 'number' > 0 but got ${JSON.stringify(e.filters)}`)}}class js extends Hs{constructor(e){super(2,e),js.verifyArgs(e)}getConfig(){const e=super.getConfig();return delete e.rank,e}static verifyArgs(e){if(typeof e.kernelSize!="number"&&!ol(e.kernelSize,"number",1,2))throw new W(`Conv2D expects config.kernelSize to be number or number[] with length 1 or 2, but received ${JSON.stringify(e.kernelSize)}.`)}}js.className="Conv2D";re(js);class Rr extends Hs{constructor(e){super(3,e),Rr.verifyArgs(e)}getConfig(){const e=super.getConfig();return delete e.rank,e}static verifyArgs(e){if(typeof e.kernelSize!="number"&&!(Array.isArray(e.kernelSize)&&(e.kernelSize.length===1||e.kernelSize.length===3)))throw new W(`Conv3D expects config.kernelSize to be number or [number, number, number], but received ${JSON.stringify(e.kernelSize)}.`)}}Rr.className="Conv3D";re(Rr);class sd extends js{constructor(e){if(super(e),this.inputSpec=[new wn({ndim:4})],this.padding!=="same"&&this.padding!=="valid")throw new W(`Conv2DTranspose currently supports only padding modes 'same' and 'valid', but received padding mode ${this.padding}`)}build(e){if(e=rn(e),e.length!==4)throw new W("Input should have rank 4; Received input shape: "+JSON.stringify(e));const t=this.dataFormat==="channelsFirst"?1:e.length-1;if(e[t]==null)throw new W("The channel dimension of the inputs should be defined. Found `None`.");const n=e[t],r=this.kernelSize.concat([this.filters,n]);this.kernel=this.addWeight("kernel",r,"float32",this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight("bias",[this.filters],"float32",this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint)),this.inputSpec=[new wn({ndim:4,axes:{[t]:n}})],this.built=!0}call(e,t){return Y(()=>{let n=kt(e);if(n.shape.length!==4)throw new W(`Conv2DTranspose.call() expects input tensor to be rank-4, but received a tensor of rank-${n.shape.length}`);const r=n.shape,i=r[0];let o,a;this.dataFormat==="channelsFirst"?(o=2,a=3):(o=1,a=2);const l=r[o],u=r[a],c=this.kernelSize[0],h=this.kernelSize[1],d=this.strides[0],w=this.strides[1],I=gn(l,d,c,this.padding),E=gn(u,w,h,this.padding),m=[i,I,E,this.filters];this.dataFormat!=="channelsLast"&&(n=Be(n,[0,2,3,1]));let S=q0(n,this.kernel.read(),m,this.strides,this.padding);return this.dataFormat!=="channelsLast"&&(S=Be(S,[0,3,1,2])),this.bias!=null&&(S=Mr(S,this.bias.read(),this.dataFormat)),this.activation!=null&&(S=this.activation.apply(S)),S})}computeOutputShape(e){e=rn(e);const t=e.slice();let n,r,i;this.dataFormat==="channelsFirst"?(n=1,r=2,i=3):(n=3,r=1,i=2);const o=this.kernelSize[0],a=this.kernelSize[1],l=this.strides[0],u=this.strides[1];return t[n]=this.filters,t[r]=gn(t[r],l,o,this.padding),t[i]=gn(t[i],u,a,this.padding),t}getConfig(){const e=super.getConfig();return delete e.dilationRate,e}}sd.className="Conv2DTranspose";re(sd);class rd extends Rr{constructor(e){if(super(e),this.inputSpec=[new wn({ndim:5})],this.padding!=="same"&&this.padding!=="valid")throw new W(`Conv3DTranspose currently supports only padding modes 'same' and 'valid', but received padding mode ${this.padding}`)}build(e){if(e=rn(e),e.length!==5)throw new W("Input should have rank 5; Received input shape: "+JSON.stringify(e));const t=this.dataFormat==="channelsFirst"?1:e.length-1;if(e[t]==null)throw new W("The channel dimension of the inputs should be defined. Found `None`.");const n=e[t],r=this.kernelSize.concat([this.filters,n]);this.kernel=this.addWeight("kernel",r,"float32",this.kernelInitializer,this.kernelRegularizer,!0,this.kernelConstraint),this.useBias&&(this.bias=this.addWeight("bias",[this.filters],"float32",this.biasInitializer,this.biasRegularizer,!0,this.biasConstraint)),this.inputSpec=[new wn({ndim:5,axes:{[t]:n}})],this.built=!0}call(e,t){return Y(()=>{let n=kt(e);if(n.shape.length!==5)throw new W(`Conv3DTranspose.call() expects input tensor to be rank-4, but received a tensor of rank-${n.shape.length}`);const r=n.shape,i=r[0];let o,a,l;this.dataFormat==="channelsFirst"?(l=2,o=3,a=4):(l=1,o=2,a=3);const u=r[l],c=r[o],h=r[a],d=this.kernelSize[0],w=this.kernelSize[1],I=this.kernelSize[2],E=this.strides[0],m=this.strides[1],S=this.strides[2],b=gn(u,E,d,this.padding),f=gn(c,m,w,this.padding),_=gn(h,S,I,this.padding),v=[i,b,f,_,this.filters];this.dataFormat!=="channelsLast"&&(n=Be(n,[0,2,3,4,1]));let T=Q0(n,this.kernel.read(),v,this.strides,this.padding);return this.dataFormat!=="channelsLast"&&(T=Be(T,[0,4,1,2,3])),this.bias!==null&&(T=Mr(T,this.bias.read(),this.dataFormat)),this.activation!==null&&(T=this.activation.apply(T)),T})}computeOutputShape(e){e=rn(e);const t=e.slice();let n,r,i,o;this.dataFormat==="channelsFirst"?(n=1,r=2,i=3,o=4):(n=4,r=1,i=2,o=3);const a=this.kernelSize[0],l=this.kernelSize[1],u=this.kernelSize[2],c=this.strides[0],h=this.strides[1],d=this.strides[2];return t[n]=this.filters,t[r]=gn(t[r],c,a,this.padding),t[i]=gn(t[i],h,l,this.padding),t[o]=gn(t[o],d,u,this.padding),t}getConfig(){const e=super.getConfig();return delete e.dilationRate,e}}rd.className="Conv3DTranspose";re(rd);class id extends Hs{constructor(e,t){if(super(e,t),this.DEFAULT_DEPTHWISE_INITIALIZER="glorotUniform",this.DEFAULT_POINTWISE_INITIALIZER="glorotUniform",this.depthwiseKernel=null,this.pointwiseKernel=null,t.filters==null)throw new W("The `filters` configuration field is required by SeparableConv, but is unspecified.");if(t.kernelInitializer!=null||t.kernelRegularizer!=null||t.kernelConstraint!=null)throw new W("Fields kernelInitializer, kernelRegularizer and kernelConstraint are invalid for SeparableConv2D. Use depthwiseInitializer, depthwiseRegularizer, depthwiseConstraint, pointwiseInitializer, pointwiseRegularizer and pointwiseConstraint instead.");if(t.padding!=null&&t.padding!=="same"&&t.padding!=="valid")throw new W(`SeparableConv${this.rank}D supports only padding modes: 'same' and 'valid', but received ${JSON.stringify(t.padding)}`);this.depthMultiplier=t.depthMultiplier==null?1:t.depthMultiplier,this.depthwiseInitializer=xr(t.depthwiseInitializer||this.DEFAULT_DEPTHWISE_INITIALIZER),this.depthwiseRegularizer=_r(t.depthwiseRegularizer),this.depthwiseConstraint=Ui(t.depthwiseConstraint),this.pointwiseInitializer=xr(t.depthwiseInitializer||this.DEFAULT_POINTWISE_INITIALIZER),this.pointwiseRegularizer=_r(t.pointwiseRegularizer),this.pointwiseConstraint=Ui(t.pointwiseConstraint)}build(e){if(e=rn(e),e.length<this.rank+2)throw new W(`Inputs to SeparableConv${this.rank}D should have rank ${this.rank+2}, but received input shape: ${JSON.stringify(e)}`);const t=this.dataFormat==="channelsFirst"?1:e.length-1;if(e[t]==null||e[t]<0)throw new W(`The channel dimension of the inputs should be defined, but found ${JSON.stringify(e[t])}`);const n=e[t],r=this.kernelSize.concat([n,this.depthMultiplier]),i=[];for(let a=0;a<this.rank;++a)i.push(1);i.push(n*this.depthMultiplier,this.filters);const o=!0;this.depthwiseKernel=this.addWeight("depthwise_kernel",r,"float32",this.depthwiseInitializer,this.depthwiseRegularizer,o,this.depthwiseConstraint),this.pointwiseKernel=this.addWeight("pointwise_kernel",i,"float32",this.pointwiseInitializer,this.pointwiseRegularizer,o,this.pointwiseConstraint),this.useBias?this.bias=this.addWeight("bias",[this.filters],"float32",this.biasInitializer,this.biasRegularizer,o,this.biasConstraint):this.bias=null,this.inputSpec=[new wn({ndim:this.rank+2,axes:{[t]:n}})],this.built=!0}call(e,t){return Y(()=>{e=kt(e);let n;if(this.rank===1)throw new be("1D separable convolution is not implemented yet.");return this.rank===2&&(this.dataFormat==="channelsFirst"&&(e=Be(e,[0,2,3,1])),n=Rb(e,this.depthwiseKernel.read(),this.pointwiseKernel.read(),this.strides,this.padding,this.dilationRate,"NHWC")),this.useBias&&(n=Mr(n,this.bias.read(),this.dataFormat)),this.activation!=null&&(n=this.activation.apply(n)),this.dataFormat==="channelsFirst"&&(n=Be(n,[0,3,1,2])),n})}getConfig(){const e=super.getConfig();return delete e.rank,delete e.kernelInitializer,delete e.kernelRegularizer,delete e.kernelConstraint,e.depthwiseInitializer=zi(this.depthwiseInitializer),e.pointwiseInitializer=zi(this.pointwiseInitializer),e.depthwiseRegularizer=vr(this.depthwiseRegularizer),e.pointwiseRegularizer=vr(this.pointwiseRegularizer),e.depthwiseConstraint=Fi(this.depthwiseConstraint),e.pointwiseConstraint=Fi(this.pointwiseConstraint),e}}id.className="SeparableConv";class od extends id{constructor(e){super(2,e)}}od.className="SeparableConv2D";re(od);class bo extends Hs{constructor(e){super(1,e),bo.verifyArgs(e),this.inputSpec=[{ndim:3}]}getConfig(){const e=super.getConfig();return delete e.rank,delete e.dataFormat,e}static verifyArgs(e){if(typeof e.kernelSize!="number"&&!ol(e.kernelSize,"number",1,1))throw new W(`Conv1D expects config.kernelSize to be number or number[] with length 1, but received ${JSON.stringify(e.kernelSize)}.`)}}bo.className="Conv1D";re(bo);class ad extends on{constructor(e){super(e),typeof e.cropping=="number"?this.cropping=[[e.cropping,e.cropping],[e.cropping,e.cropping]]:typeof e.cropping[0]=="number"?this.cropping=[[e.cropping[0],e.cropping[0]],[e.cropping[1],e.cropping[1]]]:this.cropping=e.cropping,this.dataFormat=e.dataFormat===void 0?"channelsLast":e.dataFormat,this.inputSpec=[{ndim:4}]}computeOutputShape(e){return this.dataFormat==="channelsFirst"?[e[0],e[1],e[2]-this.cropping[0][0]-this.cropping[0][1],e[3]-this.cropping[1][0]-this.cropping[1][1]]:[e[0],e[1]-this.cropping[0][0]-this.cropping[0][1],e[2]-this.cropping[1][0]-this.cropping[1][1],e[3]]}call(e,t){return Y(()=>{if(e=kt(e),this.dataFormat==="channelsLast"){const n=Qr(e,this.cropping[0][0],e.shape[1]-this.cropping[0][0]-this.cropping[0][1],2);return Qr(n,this.cropping[1][0],e.shape[2]-this.cropping[1][1]-this.cropping[1][0],3)}else{const n=Qr(e,this.cropping[0][0],e.shape[2]-this.cropping[0][0]-this.cropping[0][1],3);return Qr(n,this.cropping[1][0],e.shape[3]-this.cropping[1][1]-this.cropping[1][0],4)}})}getConfig(){const e={cropping:this.cropping,dataFormat:this.dataFormat},t=super.getConfig();return Object.assign(e,t),e}}ad.className="Cropping2D";re(ad);class bl extends on{constructor(e){super(e),this.DEFAULT_SIZE=[2,2],this.inputSpec=[{ndim:4}],this.size=e.size==null?this.DEFAULT_SIZE:e.size,this.dataFormat=e.dataFormat==null?"channelsLast":e.dataFormat,Ve(this.dataFormat),this.interpolation=e.interpolation==null?"nearest":e.interpolation,zx(this.interpolation)}computeOutputShape(e){if(this.dataFormat==="channelsFirst"){const t=e[2]==null?null:this.size[0]*e[2],n=e[3]==null?null:this.size[1]*e[3];return[e[0],e[1],t,n]}else{const t=e[1]==null?null:this.size[0]*e[1],n=e[2]==null?null:this.size[1]*e[2];return[e[0],t,n,e[3]]}}call(e,t){return Y(()=>{let n=kt(e);const r=n.shape;if(this.dataFormat==="channelsFirst"){n=Be(n,[0,2,3,1]);const i=this.size[0]*r[2],o=this.size[1]*r[3],a=this.interpolation==="nearest"?Xr.resizeNearestNeighbor(n,[i,o]):Xr.resizeBilinear(n,[i,o]);return Be(a,[0,3,1,2])}else{const i=this.size[0]*r[1],o=this.size[1]*r[2];return this.interpolation==="nearest"?Xr.resizeNearestNeighbor(n,[i,o]):Xr.resizeBilinear(n,[i,o])}})}getConfig(){const e={size:this.size,dataFormat:this.dataFormat,interpolation:this.interpolation},t=super.getConfig();return Object.assign(e,t),e}}bl.className="UpSampling2D";re(bl);/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function wo(s,e,t,n,r,i){return Y(()=>{Ve(r),gf(i),jt(n),t==null&&(t=[1,1]),n==null&&(n="valid"),r==null&&(r=qs()),i==null&&(i="max"),s=td(s,r);let o;const a=n==="same"?"same":"valid";return i==="max"?o=j1(s,e,t,a):o=$0(s,e,t,a),r==="channelsFirst"&&(o=Be(o,[0,3,1,2])),o})}function ld(s,e,t,n,r,i){return Y(()=>{Ve(r),gf(i),jt(n),t==null&&(t=[1,1,1]),n==null&&(n="valid"),r==null&&(r=qs()),i==null&&(i="max"),s=nd(s,r);let o;const a=n==="same"?"same":"valid";return i==="max"?o=X1(s,e,t,a):o=D0(s,e,t,a),r==="channelsFirst"&&(o=Be(o,[0,4,1,2,3])),o})}class ud extends on{constructor(e){if(e.poolSize==null&&(e.poolSize=2),super(e),typeof e.poolSize=="number")this.poolSize=[e.poolSize];else if(Array.isArray(e.poolSize)&&e.poolSize.length===1&&typeof e.poolSize[0]=="number")this.poolSize=e.poolSize;else throw new W(`poolSize for 1D convolutional layer must be a number or an Array of a single number, but received ${JSON.stringify(e.poolSize)}`);if($n(this.poolSize,"poolSize"),e.strides==null)this.strides=this.poolSize;else if(typeof e.strides=="number")this.strides=[e.strides];else if(Array.isArray(e.strides)&&e.strides.length===1&&typeof e.strides[0]=="number")this.strides=e.strides;else throw new W(`strides for 1D convolutional layer must be a number or an Array of a single number, but received ${JSON.stringify(e.strides)}`);$n(this.strides,"strides"),this.padding=e.padding==null?"valid":e.padding,jt(this.padding),this.inputSpec=[new wn({ndim:3})]}computeOutputShape(e){e=rn(e);const t=rs(e[1],this.poolSize[0],this.padding,this.strides[0]);return[e[0],t,e[2]]}call(e,t){return Y(()=>{this.invokeCallHook(e,t),e=al(kt(e),2);const n=this.poolingFunction(kt(e),[this.poolSize[0],1],[this.strides[0],1],this.padding,"channelsLast");return yo(n,[2])})}getConfig(){const e={poolSize:this.poolSize,padding:this.padding,strides:this.strides},t=super.getConfig();return Object.assign(e,t),e}}class cd extends ud{constructor(e){super(e)}poolingFunction(e,t,n,r,i){return Ve(i),jt(r),wo(e,t,n,r,i,"max")}}cd.className="MaxPooling1D";re(cd);class hd extends ud{constructor(e){super(e)}poolingFunction(e,t,n,r,i){return Ve(i),jt(r),wo(e,t,n,r,i,"avg")}}hd.className="AveragePooling1D";re(hd);class fd extends on{constructor(e){if(e.poolSize==null&&(e.poolSize=[2,2]),super(e),this.poolSize=Array.isArray(e.poolSize)?e.poolSize:[e.poolSize,e.poolSize],e.strides==null)this.strides=this.poolSize;else if(Array.isArray(e.strides)){if(e.strides.length!==2)throw new W(`If the strides property of a 2D pooling layer is an Array, it is expected to have a length of 2, but received length ${e.strides.length}.`);this.strides=e.strides}else this.strides=[e.strides,e.strides];$n(this.poolSize,"poolSize"),$n(this.strides,"strides"),this.padding=e.padding==null?"valid":e.padding,this.dataFormat=e.dataFormat==null?"channelsLast":e.dataFormat,Ve(this.dataFormat),jt(this.padding),this.inputSpec=[new wn({ndim:4})]}computeOutputShape(e){e=rn(e);let t=this.dataFormat==="channelsFirst"?e[2]:e[1],n=this.dataFormat==="channelsFirst"?e[3]:e[2];return t=rs(t,this.poolSize[0],this.padding,this.strides[0]),n=rs(n,this.poolSize[1],this.padding,this.strides[1]),this.dataFormat==="channelsFirst"?[e[0],e[1],t,n]:[e[0],t,n,e[3]]}call(e,t){return Y(()=>(this.invokeCallHook(e,t),this.poolingFunction(kt(e),this.poolSize,this.strides,this.padding,this.dataFormat)))}getConfig(){const e={poolSize:this.poolSize,padding:this.padding,strides:this.strides,dataFormat:this.dataFormat},t=super.getConfig();return Object.assign(e,t),e}}class wl extends fd{constructor(e){super(e)}poolingFunction(e,t,n,r,i){return Ve(i),jt(r),wo(e,t,n,r,i,"max")}}wl.className="MaxPooling2D";re(wl);class dd extends fd{constructor(e){super(e)}poolingFunction(e,t,n,r,i){return Ve(i),jt(r),wo(e,t,n,r,i,"avg")}}dd.className="AveragePooling2D";re(dd);class pd extends on{constructor(e){if(e.poolSize==null&&(e.poolSize=[2,2,2]),super(e),this.poolSize=Array.isArray(e.poolSize)?e.poolSize:[e.poolSize,e.poolSize,e.poolSize],e.strides==null)this.strides=this.poolSize;else if(Array.isArray(e.strides)){if(e.strides.length!==3)throw new W(`If the strides property of a 3D pooling layer is an Array, it is expected to have a length of 3, but received length ${e.strides.length}.`);this.strides=e.strides}else this.strides=[e.strides,e.strides,e.strides];$n(this.poolSize,"poolSize"),$n(this.strides,"strides"),this.padding=e.padding==null?"valid":e.padding,this.dataFormat=e.dataFormat==null?"channelsLast":e.dataFormat,Ve(this.dataFormat),jt(this.padding),this.inputSpec=[new wn({ndim:5})]}computeOutputShape(e){e=rn(e);let t=this.dataFormat==="channelsFirst"?e[2]:e[1],n=this.dataFormat==="channelsFirst"?e[3]:e[2],r=this.dataFormat==="channelsFirst"?e[4]:e[3];return t=rs(t,this.poolSize[0],this.padding,this.strides[0]),n=rs(n,this.poolSize[1],this.padding,this.strides[1]),r=rs(r,this.poolSize[2],this.padding,this.strides[2]),this.dataFormat==="channelsFirst"?[e[0],e[1],t,n,r]:[e[0],t,n,r,e[4]]}call(e,t){return Y(()=>(this.invokeCallHook(e,t),this.poolingFunction(kt(e),this.poolSize,this.strides,this.padding,this.dataFormat)))}getConfig(){const e={poolSize:this.poolSize,padding:this.padding,strides:this.strides,dataFormat:this.dataFormat},t=super.getConfig();return Object.assign(e,t),e}}class md extends pd{constructor(e){super(e)}poolingFunction(e,t,n,r,i){return Ve(i),jt(r),ld(e,t,n,r,i,"max")}}md.className="MaxPooling3D";re(md);class gd extends pd{constructor(e){super(e)}poolingFunction(e,t,n,r,i){return Ve(i),jt(r),ld(e,t,n,r,i,"avg")}}gd.className="AveragePooling3D";re(gd);class yd extends on{constructor(e){super(e),this.inputSpec=[new wn({ndim:3})]}computeOutputShape(e){return[e[0],e[2]]}call(e,t){throw new be}}class bd extends yd{constructor(e){super(e||{})}call(e,t){return Y(()=>{const n=kt(e);return Xe(n,1)})}}bd.className="GlobalAveragePooling1D";re(bd);class wd extends yd{constructor(e){super(e||{})}call(e,t){return Y(()=>{const n=kt(e);return Fn(n,1)})}}wd.className="GlobalMaxPooling1D";re(wd);class xd extends on{constructor(e){super(e),this.dataFormat=e.dataFormat==null?"channelsLast":e.dataFormat,Ve(this.dataFormat),this.inputSpec=[new wn({ndim:4})]}computeOutputShape(e){return e=e,this.dataFormat==="channelsLast"?[e[0],e[3]]:[e[0],e[1]]}call(e,t){throw new be}getConfig(){const e={dataFormat:this.dataFormat},t=super.getConfig();return Object.assign(e,t),e}}class vd extends xd{call(e,t){return Y(()=>{const n=kt(e);return this.dataFormat==="channelsLast"?Xe(n,[1,2]):Xe(n,[2,3])})}}vd.className="GlobalAveragePooling2D";re(vd);class _d extends xd{call(e,t){return Y(()=>{const n=kt(e);return this.dataFormat==="channelsLast"?Fn(n,[1,2]):Fn(n,[2,3])})}}_d.className="GlobalMaxPooling2D";re(_d);/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function Wi(s,e){return Y(()=>{s.dtype!=="float32"&&(s=Ee(s,"float32"));const t=Se(Or(s),e,!0),n=mo(t.shape,ze()),r=sn(zs(t,n));return ge(s,r)})}function xo(s,e){return Y(()=>Xe(Or(we(e,s)),-1))}function xl(s,e){return Y(()=>Xe(dt(we(e,s)),-1))}function vl(s,e){return Y(()=>{const t=we(s,e),n=nn(dt(s),ze(),Number.MAX_VALUE),r=dt(ge(t,n));return J(100,Xe(r,-1))})}function y2(s,e){return Y(()=>{const t=nn(e,ze(),Number.MAX_VALUE),n=hs(ae(1,t)),r=nn(s,ze(),Number.MAX_VALUE),i=hs(ae(1,r));return Xe(Or(we(n,i)),-1)})}function b2(s,e){return Y(()=>{const t=zs(0,we(1,J(s,e)));return Xe(Or(t),-1)})}function w2(s,e){return Y(()=>{const t=zs(0,we(1,J(s,e)));return Xe(t,-1)})}function x2(s,e){return Y(()=>{const t=Se(J(s,e),-1),n=Fn(J(we(1,s),e),-1);return zs(0,ae(1,we(n,t)))})}function v2(s,e){return Y(()=>{const t=Math.log(2),n=we(e,s),r=we(ae(n,Za(J(-2,n))),t);return Xe(r,-1)})}function Sr(s,e,t=!1){return Y(()=>{if(t)e=sf(e);else{const n=Se(e,e.shape.length-1,!0);e=ge(e,n)}return e=nn(e,ze(),1-ze()),Us(Se(J(Ee(s,"float32"),hs(e)),e.shape.length-1))})}function qi(s,e,t=!1){return Y(()=>{const n=Ee(I1(qx(s)),"int32");e=nn(e,ze(),1-ze());const r=e.shape,i=se(eb(n,r[r.length-1]),r);return Sr(i,e,t)})}function _2(s,e){if(!Ht(s.shape,e.shape))throw new W(`logits and labels must have the same shape, but got shapes ${JSON.stringify(s.shape)} and ${JSON.stringify(e.shape)}`);return Y(()=>{const t=Nr(e),n=Us(dt(e));return ae(we(t,J(e,s)),B1(xa(n)))})}function vo(s,e){return Y(()=>{let t;return t=nn(e,ze(),1-ze()),t=hs(ge(t,we(1,t))),Xe(_2(s,t),-1)})}function S2(s,e){return Y(()=>{const t=nn(s,ze(),1),n=nn(e,ze(),1);return Se(J(s,hs(ge(t,n))),-1)})}function I2(s,e){return Y(()=>{const t=hs(ae(ze(),e));return Xe(we(e,J(s,t)),-1)})}function Sd(s,e){return Y(()=>{const t=Wi(s,-1),n=Wi(e,-1),r=J(t,n);return Us(Se(r,-1))})}const Hi={meanSquaredError:xo,meanAbsoluteError:xl,meanAbsolutePercentageError:vl,meanSquaredLogarithmicError:y2,squaredHinge:b2,hinge:w2,categoricalHinge:x2,logcosh:v2,categoricalCrossentropy:Sr,sparseCategoricalCrossentropy:qi,binaryCrossentropy:vo,kullbackLeiblerDivergence:S2,poisson:I2,cosineProximity:Sd};function Ho(s){if(typeof s=="string"){if(s in Hi)return Hi[s];let e=`Unknown loss ${s}`;throw s.toLowerCase().includes("softmaxcrossentropy")&&(e=`Unknown loss ${s}. Use "categoricalCrossentropy" as the string name for tf.losses.softmaxCrossEntropy`),new W(e)}else return s}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */class gs extends on{constructor(e){super(e||{}),this.supportsMasking=!0}mergeFunction(e){throw new be}computeElementwiseOpOutputShape(e,t){if(e==null||t==null)return null;if(e.length<t.length)return this.computeElementwiseOpOutputShape(t,e);if(t.length===0)return e;const n=e.slice(0,e.length-t.length);for(let r=0;r<t.length;++r){const i=e[e.length-t.length+r],o=t[r];if(i==null||o==null||i<0||o<0)n.push(null);else if(i===1)n.push(o);else if(o===1)n.push(i);else{if(i!==o)throw new W("Operands could not be broadcast together with shapes "+JSON.stringify(e)+" "+JSON.stringify(t));n.push(i)}}return n}build(e){if(Array.isArray(e)&&!Array.isArray(e[0])&&(e=[rn(e)]),e=e,e.length<2)throw new W(`A merge layer should be called on an Array of at least 2 inputs. Got ${e.length} input(s).`);let t=[];for(const i of e)i!=null&&i[0]!==null&&t.push(i[0]);if(t=ns(t),t.length>1)throw new W(`Can not merge tensors with different batch sizes. Got tensors with shapes: ${JSON.stringify(e)}.`);let n=e[0]==null?null:e[0].slice(1);for(let i=1;i<e.length;++i){const o=e[i]==null?null:e[i].slice(1);n=this.computeElementwiseOpOutputShape(n,o)}const r=e.map(i=>i.length);e.indexOf(null)===-1&&ns(r).length===1?this.reshapeRequired=!1:this.reshapeRequired=!0}call(e,t){return Y(()=>{if(e=e,this.reshapeRequired){const n=[],r=e.map(i=>i.rank);if(r.indexOf(null)===-1){const i=xf(r);for(let o of e){const a=o.rank;for(let l=0;l<i-a;++l)o=al(o,1);n.push(o)}return this.mergeFunction(n)}else{let i=!1;for(const l of e){const u=l.rank;if(u==null){const c=l.shape,h=c[0],d=c.slice(1).concat([h]);let w=se(l,[h].concat(hr(c.slice(1))));w=Be(w,[1,0]),w=se(w,d),n.push(w),i=!0}else if(u>1){const c=Bi(1,u).concat([0]);n.push(Be(l,c)),i=!0}else n.push(l)}let o=this.mergeFunction(n);const a=o.rank;if(i){if(a==null){const l=o.shape,u=l.length,c=l[u-1],h=[c].concat(l.slice(0,l.length-1));o=se(Be(se(o,[-1,c]),[1,0]),h)}else if(a>1){const l=[a-1].concat(Bi(0,a-1));o=Be(o,l)}}return o}}else return this.mergeFunction(e)})}computeOutputShape(e){e=e;let t;e[0]==null?t=null:t=e[0].slice(1);for(let r=1;r<e.length;++r){const i=e[r]==null?null:e[r].slice(1);t=this.computeElementwiseOpOutputShape(t,i)}let n=[];for(const r of e)r!=null&&r[0]!==null&&n.push(r[0]);return n=ns(n),n.length===1?t=n.concat(t):t=[null].concat(t),t}computeMask(e,t){return Y(()=>{if(t==null)return null;if(!Array.isArray(t))throw new W("`mask` should be an Array");if(!Array.isArray(e))throw new W("`inputs` should be an Array");if(t.length!==e.length)throw new W(`The Array 'inputs' and 'mask' are expected to have the same length, but have different lengths (${e.length} vs ${t.length})`);if(t.every(r=>r==null))return null;t=t.map(r=>r==null?r:dn(r,0));let n=t[0];for(let r=1;r<t.length-1;++r)n=go(n,t[r]);return n})}}class Id extends gs{constructor(e){super(e)}mergeFunction(e){return Y(()=>{let t=e[0].clone();for(let n=1;n<e.length;++n)t=ae(t,e[n]);return t})}}Id.className="Add";re(Id);class kd extends gs{constructor(e){super(e)}mergeFunction(e){return Y(()=>{let t=e[0].clone();for(let n=1;n<e.length;++n)t=J(t,e[n]);return t})}}kd.className="Multiply";re(kd);class Td extends gs{constructor(e){super(e)}mergeFunction(e){return Y(()=>{let t=e[0].clone();for(let n=1;n<e.length;++n)t=ae(t,e[n]);return J(1/e.length,t)})}}Td.className="Average";re(Td);class Ed extends gs{constructor(e){super(e)}mergeFunction(e){return Y(()=>{let t=e[0];for(let n=1;n<e.length;++n)t=zs(t,e[n]);return t})}}Ed.className="Maximum";re(Ed);class Ad extends gs{constructor(e){super(e)}mergeFunction(e){return Y(()=>{let t=e[0];for(let n=1;n<e.length;++n)t=Mi(t,e[n]);return t})}}Ad.className="Minimum";re(Ad);class _l extends gs{constructor(e){super(e),this.DEFAULT_AXIS=-1,e==null&&(e={}),this.axis=e.axis==null?this.DEFAULT_AXIS:e.axis,this.supportsMasking=!0,this.reshapeRequired=!1}build(e){if(!(Array.isArray(e)&&Array.isArray(e[0]))||e.length===1)throw new W("A `Concatenate` layer should be called on a list of at least 2 inputs");e=e;let t=!0;for(const r of e)if(r!=null){t=!1;break}if(t)return;const n=[];for(let r=0;r<e.length;++r){const i=e[r].slice();i.splice(this.axis,1);let o=!1;for(const a of n)if(Ht(a,i)){o=!0;break}o||n.push(i)}if(n.length>1)throw new W("A `Concatenate` layer requires inputs with matching shapes except for the concat axis. Got input shapes: "+JSON.stringify(e))}mergeFunction(e){return Y(()=>Hx(e,this.axis))}computeOutputShape(e){if(!(Array.isArray(e)&&Array.isArray(e[0])))throw new W("A `Concatenate` layer should be called on a list of inputs.");const t=e,n=t[0].slice(),r=this.axis<0?n.length+this.axis:this.axis;for(const i of t.slice(1)){if(n[r]==null||i[r]==null){n[r]=null;break}n[r]+=i[r]}return n}computeMask(e,t){if(t==null)return null;if(!Array.isArray(t))throw new W("`mask` should be an array for Concatenate");if(!Array.isArray(e))throw new W("`inputs` should be an array for Concatenate");if(t.length!==e.length)throw new W(`Mismatch in the length of mask (${t.length}) and the legnth of inputs (${e.length})`);return Y(()=>{let n=!0;if(t.forEach(o=>{if(o!=null){n=!1;return}}),n)return null;const r=[];for(let o=0;o<e.length;++o)t[o]==null?r.push(Ee(Zh(e[o]),"bool")):t[o].rank<e[o].rank?r.push(dn(t[o],-1)):r.push(t[o]);const i=es(r,this.axis);return v0(i,-1,!1)})}getConfig(){const e={axis:this.axis},t=super.getConfig();return Object.assign(e,t),e}}_l.className="Concatenate";re(_l);function Js(s,e){for(;s<0;)s+=e;return s}function k2(s,e,t){if(s.shape.length>3||e.shape.length>3)throw new be("batchDot is not implemented for tensors of 4D or higher rank yet");if(P(s.shape.length>=2,()=>`batchDot requires the rank of x to be >= 2, but got ${s.shape.length}`),P(s.shape.length>=2,()=>`batchDot requires the rank of y to be >= 2, but got ${e.shape.length}`),typeof t=="number"&&(t=[t,t]),s.dtype==="complex64"||e.dtype==="complex64")throw new be("batchDot is not implemented for complex64-type Tensors yet.");const n=s.shape.length,r=e.shape.length;t==null&&(t=[n-1,r-2]);const i=t;return Y(()=>{let o;if(n>r){o=n-r;const l=[];for(let u=0;u<o;++u)l.push(1);e=se(e,e.shape.concat(l))}else if(r>n){o=r-n;const l=[];for(let u=0;u<o;++u)l.push(1);s=se(s,s.shape.concat(l))}else o=0;let a;if(s.shape.length===2&&e.shape.length===2)i[0]===i[1]?a=Se(J(s,e),i[0]):a=Se(J(Be(s,[1,0]),e),i[1]);else{const l=i[0]!==s.shape.length-1,u=i[1]===e.shape.length-1;a=ln(s,e,l,u)}if(o>0){let l;n>r?l=n+r-3:l=n-1;const u=[];for(let c=l;c<l+o;++c)u.push(c);a=yo(a,u)}return a.shape.length===1&&(a=dn(a,1)),a})}class Cd extends gs{constructor(e){super(e),this.axes=e.axes,this.normalize=e.normalize==null?!1:e.normalize,this.supportsMasking=!0,this.reshapeRequired=!1}build(e){P(Array.isArray(e)&&e.length===2&&Array.isArray(e[0])&&Array.isArray(e[1]),()=>"A `Dot` layer should be called on a list of exactly 2 inputs.");const t=e[0],n=e[1];if(t.length>3||n.length>3)throw new be("Dot layer does not support tensors of 4D or higher rank yet.");const r=this.interpretAxes(t,n);if(t[r[0]]!==n[r[1]])throw new W(`Dimension incompatibility: ${t[r[0]]} !== ${n[r[1]]}`)}mergeFunction(e){if(e.length!==2)throw new W(`A \`Dot\` layer must be called on exactly 2 inputs, but received ${e.length} input(s).`);let t=e[0],n=e[1],r;return Array.isArray(this.axes)?r=this.axes.map((i,o)=>Js(i,e[o].shape.length)):r=[Js(this.axes,t.shape.length),Js(this.axes,n.shape.length)],this.normalize&&(t=Wi(t,r[0]),n=Wi(n,r[1])),k2(t,n,r)}interpretAxes(e,t){let n;return Array.isArray(this.axes)?n=this.axes:n=[Js(this.axes,e.length),Js(this.axes,t.length)],n}computeOutputShape(e){P(Array.isArray(e)&&e.length===2&&Array.isArray(e[0])&&Array.isArray(e[1]),()=>"A `Dot` layer should be called on a list of exactly 2 inputs.");const t=e[0].slice(),n=e[1].slice();if(t.length>3||n.length>3)throw new be("Dot layer does not support tensors of 4D or higher rank yet.");const r=this.interpretAxes(t,n);t.splice(r[0],1),n.splice(r[1],1),n.splice(0,1);const i=t.concat(n);return i.length===1&&i.push(1),i}computeMask(e,t){return null}getConfig(){const e={axes:this.axes,normalize:this.normalize},t=super.getConfig();return Object.assign(e,t),e}}Cd.className="Dot";re(Cd);/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */async function Hn(s){if(s==null)return;const e=[],t=[],n=[];for(const r in s){const i=s[r];if(typeof i!="number"){const o=i;e.push(o.data()),t.push(r),n.push(o)}}if(e.length>0){const r=await Promise.all(e);for(let i=0;i<r.length;++i)s[t[i]]=r[i][0];Ce(n)}}function $d(s){if(s!=null)for(const e in s){const t=s[e];typeof t!="number"&&t.dispose()}}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */var Wu;(function(s){s[s.SILENT=0]="SILENT",s[s.VERBOSE=1]="VERBOSE"})(Wu||(Wu={}));const T2=125;class Ir{constructor(){this.validationData=null}setParams(e){this.params=e}async onEpochBegin(e,t){}async onEpochEnd(e,t){}async onBatchBegin(e,t){}async onBatchEnd(e,t){}async onTrainBegin(e){}async onTrainEnd(e){}setModel(e){}}class E2{constructor(e,t=10){e==null&&(e=[]),this.callbacks=e,this.queueLength=t}append(e){this.callbacks.push(e)}setParams(e){for(const t of this.callbacks)t.setParams(e)}setModel(e){for(const t of this.callbacks)t.setModel(e)}async onEpochBegin(e,t){t==null&&(t={});for(const n of this.callbacks)await n.onEpochBegin(e,t)}async onEpochEnd(e,t){t==null&&(t={});for(const n of this.callbacks)await n.onEpochEnd(e,t)}async onBatchBegin(e,t){t==null&&(t={});for(const n of this.callbacks)await n.onBatchBegin(e,t)}async onBatchEnd(e,t){t==null&&(t={});for(const n of this.callbacks)await n.onBatchEnd(e,t)}async onTrainBegin(e){e==null&&(e={});for(const t of this.callbacks)await t.onTrainBegin(e)}async onTrainEnd(e){e==null&&(e={});for(const t of this.callbacks)await t.onTrainEnd(e)}}class A2 extends Ir{constructor(){super()}async onEpochBegin(e){this.seen=0,this.totals={}}async onBatchEnd(e,t){t==null&&(t={});const n=t.size==null?0:t.size;this.seen+=n;for(const r in t){const i=t[r];if(typeof i=="number")this.totals.hasOwnProperty(r)||(this.totals[r]=0),this.totals[r]=this.totals[r]+i*n;else{let o;r in this.totals?o=this.totals[r]:this.totals[r]=0;const a=Y(()=>ae(this.totals[r],J(i,n)));this.totals[r]=a,o?.dispose()}}}async onEpochEnd(e,t){if(t!=null)for(const n of this.params.metrics)this.totals[n]!=null&&(typeof this.totals[n]=="number"?t[n]=this.totals[n]/this.seen:Y(()=>{const r=J(ge(1,this.seen),this.totals[n]);t[n]=r,this.totals[n].dispose(),As(t[n])}))}}class C2 extends Ir{async onTrainBegin(e){this.epoch=[],this.history={}}async onEpochEnd(e,t){t==null&&(t={}),this.epoch.push(e);for(const n in t)this.history[n]==null&&(this.history[n]=[]),this.history[n].push(t[n])}async syncData(){const e=[],t=[],n=[];for(const i in this.history){const o=this.history[i];for(let a=0;a<o.length;++a)if(typeof o[a]!="number"){const l=o[a];e.push(l.data()),t.push(i),n.push(a)}}const r=await Promise.all(e);for(let i=0;i<r.length;++i)this.history[t[i]][n[i]].dispose(),this.history[t[i]][n[i]]=r[i][0]}}class $2 extends Ir{constructor(e,t){if(super(),this.currentEpoch=0,this.nowFunc=e.nowFunc,this.nextFrameFunc=e.nextFrameFunc||ax,this.yieldEvery=t||"auto",this.yieldEvery==="auto"&&(this.yieldEvery=T2),this.yieldEvery==="never"&&e.onYield!=null)throw new Error("yieldEvery is `never` but you provided an `onYield` callback. Either change `yieldEvery` or remove the callback");oa(this.yieldEvery)&&(this.maybeWait=Fx(this.maybeWait.bind(this),this.yieldEvery,this.nowFunc)),this.trainBegin=e.onTrainBegin,this.trainEnd=e.onTrainEnd,this.epochBegin=e.onEpochBegin,this.epochEnd=e.onEpochEnd,this.batchBegin=e.onBatchBegin,this.batchEnd=e.onBatchEnd,this.yield=e.onYield}async maybeWait(e,t,n){const r=[];this.yield!=null&&(await Hn(n),r.push(this.yield(e,t,n))),r.push(this.nextFrameFunc()),await Promise.all(r)}async onEpochBegin(e,t){this.currentEpoch=e,this.epochBegin!=null&&(await Hn(t),await this.epochBegin(e,t))}async onEpochEnd(e,t){const n=[];this.epochEnd!=null&&(await Hn(t),n.push(this.epochEnd(e,t))),this.yieldEvery==="epoch"&&n.push(this.nextFrameFunc()),await Promise.all(n)}async onBatchBegin(e,t){this.batchBegin!=null&&(await Hn(t),await this.batchBegin(e,t))}async onBatchEnd(e,t){const n=[];this.batchEnd!=null&&(await Hn(t),n.push(this.batchEnd(e,t))),this.yieldEvery==="batch"?n.push(this.nextFrameFunc()):oa(this.yieldEvery)&&n.push(this.maybeWait(this.currentEpoch,e,t)),await Promise.all(n)}async onTrainBegin(e){this.trainBegin!=null&&(await Hn(e),await this.trainBegin(e))}async onTrainEnd(e){this.trainEnd!=null&&(await Hn(e),await this.trainEnd(e))}}function Nd(s,e){return s==null&&(s={}),s instanceof Ir?[s]:Array.isArray(s)&&s[0]instanceof Ir?s:ke(s).map(n=>new $2(n,e))}class Ut{constructor(){}static registerCallbackConstructor(e,t){P(e>=0&&Number.isInteger(e),()=>`Verbosity level is expected to be an integer >= 0, but got ${e}`),Ut.checkForDuplicate(t),Ut.constructors[e]==null&&(Ut.constructors[e]=[]),Ut.constructors[e].push(t)}static checkForDuplicate(e){for(const t in Ut.constructors)Ut.constructors[+t].forEach(r=>{if(r===e)throw new W("Duplicate callback constructor.")})}static clear(){Ut.constructors={}}static createCallbacks(e){const t=[];for(const n in Ut.constructors){const r=+n;e>=r&&t.push(...Ut.constructors[r])}return t.map(n=>new n)}}Ut.constructors={};function Dd(s,e,t,n,r,i,o,a,l){const u=new C2,c=[new A2,...Ut.createCallbacks(e)];s!=null&&c.push(...s),c.push(u);const h=new E2(c);return h.setParams({epochs:t,initialEpoch:n,samples:r,steps:i,batchSize:o,verbose:e,doValidation:a,metrics:l}),{callbackList:h,history:u}}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function Od(s,e={},t=!1){return Dr(s,Vt.getMap().classNameMap,e,"layer",t)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function Md(s,e){return Y(()=>{const t=J(.5,Zh(e)),n=vf($r(e,t),s.dtype);return Xe(cs(s,n),-1)})}function Pd(s,e){return Y(()=>vf(cs($i(s,-1),$i(e,-1)),"float32"))}function N2(s,e){return Y(()=>Ee(Se(go(cs(s,1),cs(e,1))),"float32"))}function D2(s,e){return Y(()=>Ee(Se(go(cs(s,0),cs(e,1))),"float32"))}function O2(s,e){return Y(()=>{const t=N2(s,e),n=D2(s,e),r=ae(t,n);return Ee(ts($r(r,0),ge(t,r),0),"float32")})}function M2(s,e){return vo(s,e)}function P2(s,e){return s.rank===e.rank&&(s=yo(s,[s.rank-1])),e=$i(e,-1),e.dtype!==s.dtype&&(e=Ee(e,s.dtype)),Ee(cs(s,e),"float32")}const R2=xo,L2=xo,B2=xl,F2=xl,U2=vl,z2=vl,Rd=Sr,V2=Sd,Ld=qi,ji={binaryAccuracy:Md,categoricalAccuracy:Pd,precision:O2,categoricalCrossentropy:Rd,sparseCategoricalCrossentropy:Ld,mse:R2,MSE:L2,mae:B2,MAE:F2,mape:U2,MAPE:z2,cosine:V2};function G2(s){if(typeof s=="string"&&s in ji)return ji[s];if(typeof s!="string"&&s!=null)return s;throw new W(`Unknown metric ${s}`)}function Jr(s){if(mn(s!==null,`Unknown LossOrMetricFn ${s}`),typeof s=="string")return s;{let e;for(const t of Object.keys(Hi))if(Hi[t]===s){e=t;break}if(e!==void 0)return e;for(const t of Object.keys(ji))if(ji[t]===s){e=t;break}return e!==void 0?e:s.name}}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function W2(s){const e={Adagrad:()=>ys.adagrad(.01),Adadelta:()=>ys.adadelta(1,.95,ze()),Adam:()=>ys.adam(.001,.9,.999,ze()),Adamax:()=>ys.adamax(.002,.9,.999,ze(),0),RMSProp:()=>ys.rmsprop(.001,.9,0,ze()),SGD:()=>ys.sgd(.01)};if(e.adagrad=e.Adagrad,e.adadelta=e.Adadelta,e.adam=e.Adam,e.adamax=e.Adamax,e.rmsprop=e.RMSProp,e.sgd=e.SGD,s in e)return e[s]();throw new W(`Unknown Optimizer ${s}`)}/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */const qu=1*1024*1024;function Hu(s,e,t=!1){if(s==null||typeof s!="object"||Object.getPrototypeOf(s)!==Object.prototype||!Sa(s))throw new Error("User-defined metadata is expected to be a JSON object, but is not.");if(t){const n=JSON.stringify(s);n.length>qu&&console.warn(`User-defined metadata of model "${e}" is too large in size (length=${n.length} when serialized). It is not recommended to store such large objects in user-defined metadata. Please make sure its serialized length is <= ${qu}.`)}}function Sa(s){if(s===null)return!0;if(typeof s=="object")if(Object.getPrototypeOf(s)===Object.prototype){const e=Object.keys(s);for(const t of e)if(typeof t!="string"||!Sa(s[t]))return!1;return!0}else if(Array.isArray(s)){for(const e of s)if(!Sa(e))return!1;return!0}else return!1;else{const e=typeof s;return e==="string"||e==="number"||e==="boolean"}}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function q2(s,e,t,n=console.log){const r=j2(s),i=["Layer (type)","Input Shape","Output shape","Param #"];r?(e=e||90,t=t||[.32,.61,.89,1]):(e=e||115,t=t||[.24,.48,.7,.8,1]),t[t.length-1]<=1&&(t=t.map(c=>Math.floor(e*c)));let o;if(!r){i.push("Receives inputs"),o=[];for(const c in s.nodesByDepth)o.push(...s.nodesByDepth[c])}n("_".repeat(e)),Ki(i,t,n),n("=".repeat(e));const a=s.layers;for(let c=0;c<a.length;++c)r?K2(a[c],t,n):X2(a[c],t,o,n),n((c===a.length-1?"=":"_").repeat(e));s.checkTrainableWeightsConsistency();const l=H2(s),u=Gi(s.nonTrainableWeights);n(`Total params: ${l+u}`),n(`Trainable params: ${l}`),n(`Non-trainable params: ${u}`),n("_".repeat(e))}function H2(s){let e;return s.collectedTrainableWeights!=null?e=Gi(s.collectedTrainableWeights):e=Gi(s.trainableWeights),e}function j2(s){let e=!0;const t=[],n=[];for(const r in s.nodesByDepth)t.push(s.nodesByDepth[r]);for(const r of t){if(r.length>1||r.length===1&&r[0].inboundLayers.length>1){e=!1;break}n.push(...r)}if(e)for(const r of s.layers){let i=!1;for(const o of r.inboundNodes)if(n.indexOf(o)!==-1)if(i){e=!1;break}else i=!0;if(!e)break}return e}function Ki(s,e,t=console.log){let n="";for(let r=0;r<s.length;++r)r>0&&(n=n.slice(0,n.length-1)+" "),n+=s[r],n=n.slice(0,e[r]),n+=" ".repeat(e[r]-n.length);t(n)}function K2(s,e,t){let n,r;try{r=s.inboundNodes.map(l=>JSON.stringify(l.inputShapes)).join(",")}catch{r="multiple"}try{n=JSON.stringify(s.outputShape)}catch{n="multiple"}const i=s.name,o=s.getClassName(),a=[`${i} (${o})`,r,n,s.countParams().toString()];Ki(a,e,t)}function X2(s,e,t,n){let r,i;try{i=s.inboundNodes.map(h=>JSON.stringify(h.inputShapes)).join(",")}catch{i="multiple"}try{r=JSON.stringify(s.outputShape)}catch{r="multiple"}const o=[];for(const h of s.inboundNodes)if(!(t!=null&&t.length>0&&t.indexOf(h)===-1))for(let d=0;d<h.inboundLayers.length;++d){const w=h.inboundLayers[d].name,I=h.nodeIndices[d],E=h.tensorIndices[d];o.push(`${w}[${I}][${E}]`)}const a=s.name,l=s.getClassName(),u=o.length===0?"":o[0],c=[`${a} (${l})`,i,r,s.countParams().toString(),u];Ki(c,e,n);for(let h=1;h<o.length;++h)Ki(["","","","",o[h]],e,n)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function Bd(s,e,t){return(s==="inboundNodes"||s==="outputLayers"||s==="inputLayers")&&e===0&&typeof t=="string"}function Ia(s,e){if(s===null)return null;if(typeof s=="string")return Yn(s);if(typeof s=="number"||typeof s=="boolean")return s;if(s instanceof Array){const t=[],n=s.length;for(let r=0;r<n;++r){const i=s[r];Bd(e,r,i)?t.push(i):t.push(Ia(i,e))}return t}else{const t={};for(const n of Object.keys(s)){const r=s[n];if(n==="name"&&typeof r=="string")t[n]=r;else{const i=Yn(n);t[i]=Ia(r,i)}}return t}}function ka(s,e){if(s==null)return null;if(typeof s=="string")return In(s);if(typeof s=="number"||typeof s=="boolean")return s;if(s instanceof Array){const t=[],n=s.length;for(let r=0;r<n;++r){const i=s[r];Bd(e,r,i)?t.push(i):t.push(ka(i,e))}return t}else{const t={};for(const n of Object.keys(s)){const r=s[n],i=In(n);(n==="name"||n==="className")&&typeof r=="string"?t[i]=r:t[i]=ka(r,n)}return t}}/** @license See the LICENSE file. */const Fd="4.20.0";/**
 * @license
 * Copyright 2022 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */class Ud{constructor(e){this.maxEntries=e||100,this.cache=new Map}get(e){let t;return this.cache.has(e)&&(t=this.cache.get(e),this.cache.delete(e),this.cache.set(e,t)),t}put(e,t){if(this.cache.has(e))this.cache.delete(e);else if(this.cache.size>=this.maxEntries){const n=this.cache.keys().next().value;this.cache.delete(n)}this.cache.set(e,t)}getMaxEntries(){return this.maxEntries}setMaxEntries(e){if(e<0)throw new Error(`The maxEntries of LRU caches must be at least 0, but got ${e}.`);if(this.maxEntries>e)for(let t=0;t<this.maxEntries-e;t++){const n=this.cache.keys().next().value;this.cache.delete(n)}this.maxEntries=e}}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */class Lr extends on{constructor(e){if(super({dtype:e.dtype,name:e.name!=null?e.name:ul("input").toString()}),e.batchSize==null&&(e.batchSize=null),e.sparse==null&&(e.sparse=!1),this.trainable=!1,this.built=!0,this.sparse=e.sparse,e.inputShape!=null&&e.batchInputShape!=null)throw new W("Only provide the inputShape OR batchInputShape argument to inputLayer, not both at the same time.");let t=e.batchInputShape;if(t==null){if(e.inputShape==null)throw new W("An InputLayer should be passed either a `batchInputShape` or an `inputShape`.");t=[e.batchSize].concat(e.inputShape)}else if(e.batchSize!=null)throw new W("Cannot specify batchSize if batchInputShape is specified when creating an InputLayer.");const n=e.dtype||"float32";this.batchInputShape=t,this.dtype=n,this.inputSpec=[{shape:t}];const r=new fs(this.dtype,this.batchInputShape,this,[],{},this.name);r.nodeIndex=0,r.tensorIndex=0,new gl({outboundLayer:this,inboundLayers:[],nodeIndices:[],tensorIndices:[],inputTensors:[r],outputTensors:[r],inputMasks:[null],outputMasks:[null],inputShapes:[t],outputShapes:[t]})}apply(e,t){throw new W(`Cannot pass any input to an InputLayer's apply() method. InputLayer name: ${this.name}`)}dispose(){return{refCountAfterDispose:this._refCount,numDisposedVariables:0}}getConfig(){return{batchInputShape:this.batchInputShape,dtype:this.dtype,sparse:this.sparse,name:this.name}}}Lr.className="InputLayer";re(Lr);function Y2(s){if(s.batchShape==null&&s.shape==null)throw new Error("Please provide to Input either a `shape` or a `batchShape` argument. Note that `shape` does not include the batch dimension.");if(s.batchShape!=null&&s.shape!=null)throw new W("Please provide either a `shape` or `batchShape` argument to Input, but not both.");let e=s.batchShape;s.shape!=null&&e==null&&(e=[null].concat(s.shape));let t=s.dtype;return t==null&&(t="float32"),new Lr({batchInputShape:e,name:s.name,dtype:t,sparse:s.sparse}).inboundNodes[0].outputTensors[0]}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function Q2(s,e){if(s.dtype==null||s.dtype===e.dtype)return e;try{return Ee(e,s.dtype)}catch{throw new W(`The dtype of the feed (${e.dtype}) can not be cast to the dtype of the key '${s.name}' (${s.dtype}).`)}}class Rn{constructor(e){if(this.id2Value={},this.id2Mask={},this.name2Id={},e instanceof Rn)for(const t in e.id2Value)this.id2Value[t]=e.id2Value[t],t in e.id2Mask&&(this.id2Mask[t]=e.id2Mask[t]);else{if(e==null)return;for(const t of e)this.add(t.key,t.value)}}add(e,t,n){if(this.id2Value[e.id]==null)this.id2Value[e.id]=Q2(e,t),this.name2Id[e.name]=e.id,n!=null&&(this.id2Mask[e.id]=n);else throw new W(`Duplicate key: name=${e.name}, id=${e.id}`);return this}addFeed(e){this.add(e.key,e.value)}hasKey(e){return this.id2Value[e.id]!=null}names(){return Object.keys(this.name2Id)}getValue(e){if(e instanceof fs){if(this.id2Value[e.id]==null)throw new W(`Nonexistent key: ${e.name}`);return this.id2Value[e.id]}else{const t=this.name2Id[e];if(t==null)throw new W(`Feed dict has no SymbolicTensor name: ${e}`);return this.id2Value[t]}}getMask(e){if(e instanceof fs){if(this.id2Value[e.id]==null)throw new W(`Nonexistent key: ${e.name}`);return this.id2Mask[e.id]}else{const t=this.name2Id[e];if(t==null)throw new W(`Feed dict has no SymbolicTensor name: ${e}`);return this.id2Mask[t]}}disposeMasks(){this.id2Mask!=null&&Ce(this.id2Mask)}}const ju=new Ud,Ku=new Ud;function rr(s,e,t,n){const r=t==null?!1:t.training,i=Array.isArray(s),o=i?s:[s],a=o.map(I=>I.name),l=[],u=e.names();for(const I of a)u.indexOf(I)!==-1?l.push(e.getValue(I)):l.push(null);const c=a.join(",")+"|"+e.names().sort().join(",");let h=ju.get(c),d;if(h==null){const I=Z2(o,e);h=I.sorted,d=I.recipientCounts,ju.put(c,h),Ku.put(c,d)}d={},r||Object.assign(d,Ku.get(c));const w=new Rn(e);for(let I=0;I<h.length;++I){const E=h[I],m=E.sourceLayer;if(m instanceof Lr)continue;const S=[],b=[],f=[];let _=!1;for(const $ of E.inputs){const A=w.getValue($),g=w.getMask($);S.push(A),b.push(g),g!=null&&(_=!0),r||(d[$.name]--,d[$.name]===0&&!e.hasKey($)&&a.indexOf($.name)===-1&&!A.isDisposed&&$.sourceLayer.stateful!==!0&&f.push(A))}_&&(t=t||{},t.mask=b[0]);const v=ke(m.apply(S,t));let T=null;m.supportsMasking&&(T=m.computeMask(S,b));const N=ev(E),O=Array.isArray(N)?N:[N];for(let $=0;$<O.length;++$){w.hasKey(O[$])||w.add(O[$],v[$],Array.isArray(T)?T[0]:T);const A=a.indexOf(O[$].name);A!==-1&&(l[A]=v[$])}r||Ce(f)}return w.disposeMasks(),i?l:l[0]}function Z2(s,e){P(s!=null&&s.length>0,()=>"Expected at least one fetch, got none");let t=[],n={};if(s.length===1){const r=Xu(s[0],e);t=r.sorted,n=r.recipientMap}else{const r=new Set;for(const i of s){const{sorted:o,recipientMap:a}=Xu(i,e);for(const l of o)r.has(l.name)||(t.push(l),r.add(l.name));for(const l in a)n[l]==null&&(n[l]=new Set),a[l].forEach(u=>n[l].add(u))}}return{sorted:t,recipientCounts:J2(n)}}function J2(s){const e={};for(const t in s)e[t]=s[t].size;return e}function Xu(s,e){const t=new Set,n=[],r={};for(const a of e.names())t.add(a);const i=[],o=[];for(i.push(s);i.length>0;){const a=i[i.length-1];if(t.has(a.name)){i.pop();continue}const l=o[o.length-1]===i.length-1;if(a.inputs.length===0||l)i.pop(),n.push(a),t.add(a.name),l&&o.pop();else{o.push(i.length-1);for(const u of a.inputs)r[u.name]==null&&(r[u.name]=new Set),r[u.name].add(a.name),!t.has(u.name)&&i.push(u)}}return{sorted:n,recipientMap:r}}function ev(s){let e;if(s.sourceLayer.inboundNodes.length===1)e=s.sourceLayer.output;else{let t=null;for(let n=0;n<s.sourceLayer.inboundNodes.length;++n)for(const r of s.sourceLayer.inboundNodes[n].outputTensors)if(r.id===s.id){t=n;break}e=s.sourceLayer.getOutputAt(t)}return e}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */const tv=s=>{const e=Object.keys(s);if(e.length===0)return!1;const t=e[0].split("/");return!isNaN(parseInt(t[t.length-1],10))};class en extends on{constructor(e){if(super({}),this.containerNodes=new Set,this.name=e.name,this.name==null){const b=this.getClassName().toLowerCase();this.name=ul(b)}if(this.supportsMasking=!1,this.trainable_=!0,Array.isArray(e.inputs)?this.inputs=e.inputs.slice():this.inputs=[e.inputs],Array.isArray(e.outputs)?this.outputs=e.outputs.slice():this.outputs=[e.outputs],ns(this.inputs).length!==this.inputs.length)throw new W(`The list of inputs passed to the model is redundant. All inputs should only appear once. Found: ${this.inputs.map(b=>b.name)}`);ns(this.outputs).length!==this.outputs.length&&console.warn(`The list of outputs passed to the model is redundant. All outputs should only appear once. Found: ${this.outputs.map(b=>b.name)}`),this.inputLayers=[],this.inputLayersNodeIndices=[],this.inputLayersTensorIndices=[],this.outputLayers=[],this.outputLayersNodeIndices=[],this.outputLayersTensorIndices=[],this.layers=[],this.internalContainerRefs=[];for(const b of this.outputs){const f=b.sourceLayer,_=b.nodeIndex,v=b.tensorIndex;this.outputLayers.push(f),this.outputLayersNodeIndices.push(_),this.outputLayersTensorIndices.push(v)}for(const b of this.inputs){const f=b.sourceLayer,_=b.nodeIndex,v=b.tensorIndex;mn(_===0,"input layer has >1 nodes"),mn(v===0,"input layer has >1 tensors"),this.inputLayers.push(f),this.inputLayersNodeIndices.push(_),this.inputLayersTensorIndices.push(v)}this.inputNames=[],this.outputNames=[],this.feedInputShapes=[],this.feedInputNames=[],this.feedOutputNames=[];for(let b=0;b<this.inputLayers.length;b++){const f=this.inputLayers[b];if(!(f instanceof Lr))throw new TypeError(`Input layers to a LayersModel must be InputLayer objects. Received inputs: ${e.inputs}. Input ${b} (0-based) originates from layer type ${f.getClassName()}.`);this.inputNames.push(f.name),this.feedInputShapes.push(f.batchInputShape),this.feedInputNames.push(f.name)}for(const b of this.outputLayers)this.outputNames.push(b.name);this.internalInputShapes=this.inputs.map(b=>b.shape),this.internalOutputShapes=this.outputs.map(b=>b.shape);const t={},n={},r={},i={},o={},a=[],l=(b,f,_,v,T,N)=>{(v==null||T==null||N==null)&&(v=b.sourceLayer,T=b.nodeIndex,N=b.tensorIndex);const O=v.inboundNodes[T];if(_.indexOf(O)!==-1)throw new zn(`The tensor ${b.name} at layer "${v.name}" is part of a cycle.`);if(f.indexOf(O)!==-1)return;this.containerNodes.add(en.nodeKey(v,T)),v.id in o||(o[v.id]=Object.keys(o).length),_.indexOf(O)===-1&&_.push(O);const $=O.inboundLayers.length;for(let A=0;A<$;A++){const g=O.inputTensors[A],p=O.inboundLayers[A],y=O.nodeIndices[A],x=O.tensorIndices[A];l(g,f,_,p,y,x)}for(f.push(O);_.indexOf(O)>=0;)_.splice(_.indexOf(O),1);a.push(O)},u=[],c=[];for(const b of this.outputs)l(b,u,c);const h=a.slice().reverse();for(const b of h){n[b.id]=b,b.id in t||(t[b.id]=0);let f=t[b.id];const _=r[b.outboundLayer.id]==null?0:r[b.outboundLayer.id];f=Math.max(f,_),r[b.outboundLayer.id]=f,i[b.outboundLayer.id]=b.outboundLayer,t[b.id]=f;for(let v=0;v<b.inboundLayers.length;v++){const T=b.inboundLayers[v],N=b.nodeIndices[v],O=T.inboundNodes[N],$=t[O.id]==null?0:t[O.id];t[O.id]=Math.max(f+1,$),n[O.id]=O}}const d={};for(const b in t){const f=t[b];f in d||(d[f]=[]),d[f].push(n[b])}const w={};for(const b in r){const f=r[b];f in w||(w[f]=[]),w[f].push(i[b])}let I=Object.keys(w).map(b=>parseInt(b,10)).sort(Yr);this.layers=[];for(const b of I){const f=w[b];f.sort((_,v)=>{const T=o[_.id],N=o[v.id];return T<N?-1:T>N?1:0});for(const _ of f)_ instanceof en&&this.internalContainerRefs.push(_),this.layers.push(_)}this.layersByDepth=w,I=Object.keys(d).map(b=>parseInt(b,10)).sort(Yr);const E=this.inputs.slice(),m=[];for(const b of I)for(const f of d[b]){const _=f.outboundLayer;if(_!=null){for(const v of f.inputTensors)if(E.indexOf(v)===-1)throw new zn(`Graph disconnected: cannot obtain value for tensor ${v} at layer "${_.name}". The following previous layers were accessed without issue: ${m}`);for(const v of f.outputTensors)E.push(v);m.push(_.name)}}this.nodesByDepth=d;const S=this.layers.map(b=>b.name);for(const b of S){const f=S.filter(_=>_===b).length;if(f!==1)throw new zn(`The name "${b}" is used ${f} times in the model. All layer names should be unique. Layer names: `+JSON.stringify(S))}this.outboundNodes=[],this.inboundNodes=[],new gl({outboundLayer:this,inboundLayers:[],nodeIndices:[],tensorIndices:[],inputTensors:this.inputs,outputTensors:this.outputs,inputMasks:this.inputs.map(b=>null),outputMasks:this.outputs.map(b=>null),inputShapes:this.inputs.map(b=>b.shape),outputShapes:this.outputs.map(b=>b.shape)}),this.built=!0,this._refCount=1}assertNotDisposed(){if(this._refCount===0)throw new Error(`Container '${this.name}' is already disposed.`)}dispose(){this.assertNotDisposed();const e={refCountAfterDispose:null,numDisposedVariables:0};if(--this._refCount===0){for(const t of this.layers)e.numDisposedVariables+=t.dispose().numDisposedVariables;for(const t of this.internalContainerRefs)e.numDisposedVariables+=t.dispose().numDisposedVariables}return e.refCountAfterDispose=this._refCount,e}get trainable(){return this.trainable_}set trainable(e){this.layers.forEach(t=>{t._trainableWeights.forEach(n=>n.trainable=e)}),this.trainable_=e}get trainableWeights(){if(this._trainableWeights.length>0)throw new W("Container instance unexpectedly contains _trainableWeights.The trainable weights of a Container are a union of the trainable weights of its consituent Layers. Its own _trainableWeights must remain an empty Array.");if(!this.trainable)return[];let e=[];for(const t of this.layers)e=e.concat(t.trainableWeights);return e}get nonTrainableWeights(){const e=[];for(const t of this.layers)e.push(...t.nonTrainableWeights);if(!this.trainable){const t=[];for(const n of this.layers)t.push(...n.trainableWeights);return t.concat(e)}return e}get weights(){return this.trainableWeights.concat(this.nonTrainableWeights)}loadWeights(e,t=!0){const n={};let r=0;const i=tv(e);i&&this.parseWeights(e);for(const a of this.layers)for(const[l,u]of a.weights.entries()){const c=i?`${u.name.split("/").slice(0,-1).join("/")+"/"}${l}`:u.originalName;if(n[c]!=null)throw new W(`Duplicate weight name: ${c}`);n[c]=u,r++}const o=[];for(const a in e){let l=a;if(n[a]==null){const u=a.split("/");l=u.slice(0,-2).concat([u[u.length-1]]).join("/")}if(n[l]!=null)o.push([n[l],e[a]]);else if(t)throw new W(`Provided weight data has no target variable: ${a}`);delete n[l]}if(t){const a=[];for(const l in n)a.push(l);if(a.length>0)throw new W(`${a.length} of ${r} weights are not set: ${a}`)}Zf(o)}parseWeights(e){for(const t in Object.keys(e)){const n=t.split("/"),r=["vars","layer_checkpoint_dependencies"],i=n.map(o=>o.startsWith("_")?o.slice(1):o).filter(o=>!r.includes(o)).join("/");i!==t&&(e[i]=e[t],delete e[t])}}updatedConfig(){const e=this.getConfig(),t={};return t.className=this.getClassName(),t.config=e,t.kerasVersion=`tfjs-layers ${Fd}`,t.backend="TensorFlow.js",t}toJSON(e,t=!0){const n=ka(this.updatedConfig());return t?JSON.stringify(n):n}call(e,t){return Y(()=>{e=ke(e);const n=new Rn;for(let r=0;r<this.inputs.length;++r)n.add(this.inputs[r],e[r]);return rr(this.outputs,n,t)})}computeMask(e,t){return Y(()=>{e=ke(e);let n;return t==null?n=Li(null,e.length):n=ke(t),this.runInternalGraph(e,n)[1]})}computeOutputShape(e){const t=Vi(e);if(t.length!==this.inputLayers.length)throw new W(`Invalid inputShape argument ${e}: model has ${this.inputLayers.length} tensor inputs.`);const n={};for(let a=0;a<t.length;a++){const l=this.inputLayers[a],u=t[a],c=l.name+"_0_0";n[c]=u}const r=Object.keys(this.nodesByDepth).map(a=>parseInt(a,10)).sort(Yr);if(r.length>1)for(const a of r){const l=this.nodesByDepth[a];for(const u of l){const c=u.outboundLayer;if(this.inputLayers.map(E=>E.id).indexOf(c.id)!==-1)continue;const h=[];for(let E=0;E<u.inboundLayers.length;E++){const m=u.inboundLayers[E],S=u.nodeIndices[E],b=u.tensorIndices[E],f=`${m.name}_${S}_${b}`,_=n[f];h.push(_)}const d=c.computeOutputShape(_t(h)),w=Vi(d),I=c.inboundNodes.indexOf(u);for(let E=0;E<w.length;E++){const m=`${c.name}_${I}_${E}`;n[m]=w[E]}}}const i=[],o=[];for(let a=0;a<this.outputLayers.length;a++){const l=this.outputLayers[a],u=this.outputLayersNodeIndices[a],c=this.outputLayersTensorIndices[a],h=`${l.name}_${u}_${c}`;o.push(h)}for(let a=0;a<o.length;a++){const l=o[a];mn(l in n),i.push(n[l])}return _t(i)}runInternalGraph(e,t){t==null&&(t=Li(null,e.length));const n={};for(let l=0;l<this.inputs.length;++l){const u=this.inputs[l],c=e[l],h=t[l];n[u.id]=[c,h]}const r=Object.keys(this.nodesByDepth).map(l=>parseInt(l,10)).sort(Yr);for(const l of r){const u=this.nodesByDepth[l];for(const c of u){const h=c.outboundLayer,d=c.inputTensors,w=c.outputTensors,I=new Array;for(const E of d)E.id in n&&I.push(n[E.id]);if(I.length===d.length){let E={},m,S,b,f;if(c.callArgs!=null&&(E=c.callArgs),I.length===1){const[_,v]=I[0];E.mask==null&&(E.mask=v),b=ke(h.call(_,E)),f=ke(h.computeMask(_,v)),m=[_],S=[v]}else m=I.map(_=>_[0]),S=I.map(_=>_[1]),E.mask==null&&(E.mask=S),b=ke(h.call(m,E)),f=ke(h.computeMask(m,S));if(h.activityRegularizer)throw new be("LayersModel invocation with concrete Tensor value(s) in the presence of activity regularizer(s) is not supported yet.");for(let _=0;_<w.length;++_){const v=w[_],T=b[_],N=f[_];n[v.id]=[T,N]}}}}const i=[],o=[],a=[];for(const l of this.outputs){mn(l.id in n,`Could not compute output ${l.name} : ${l.id}`);const[u,c]=n[l.id];a.push(u.shape),i.push(u),o.push(c)}return[i,o,a]}buildNodeConversionMap(e){const t={};let n;for(const r of this.layers){n=r instanceof en?1:0;for(let i=0;i<r.inboundNodes.length;i++){const o=en.nodeKey(r,i);this.containerNodes.has(o)&&(t[o]=n,n+=1)}}return t}getLayer(e,t){if(t!=null)return this.findLayer(t);if(e==null)throw new W("Provide either a layer name or layer index");if(typeof e=="number")return this.findLayer(e);for(const n of this.layers)if(n.name===e)return n;throw new W(`No such layer: ${e}`)}findLayer(e){if(this.layers.length<=e)throw new W(`Was asked to retrieve layer at index ${e}, but model only has ${this.layers.length} layer(s).`);return this.layers[e]}calculateLosses(){return Y(()=>{const e=[];for(const t of this.layers)for(let n=0;n<t.inboundNodes.length;++n){const r=en.nodeKey(t,n);this.containerNodes.has(r)&&e.push(...t.calculateLosses())}return e})}getConfig(){const e={name:this.name},t=this.buildNodeConversionMap(this.layers),n=[];for(const o of this.layers){const a=o.getClassName(),l=o.getConfig(),u=[];for(let h=0;h<o.inboundNodes.length;h++){const d=o.inboundNodes[h],w=en.nodeKey(o,h);let I={};if(this.containerNodes.has(w)){if(d.callArgs)try{JSON.stringify(d.callArgs),I=d.callArgs}catch{console.warn(`Layer ${o.name} was passed non-serializable keyword arguments: ${d.callArgs}. They will not be included in the serialized model (and thus will be missing at deserialization time).`),I={}}if(d.inboundLayers.length>0){const E=[];for(let m=0;m<d.inboundLayers.length;m++){const S=d.inboundLayers[m],b=d.nodeIndices[m],f=d.tensorIndices[m],_=en.nodeKey(S,b);let v=t[_];v==null&&(v=0),E.push([S.name,v,f,I])}u.push(E)}}}const c={};c.name=o.name,c.className=a,c.config=l,c.inboundNodes=u,n.push(c)}e.layers=n;const r=[];for(let o=0;o<this.inputLayers.length;o++){const a=this.inputLayers[o],l=this.inputLayersNodeIndices[o],u=en.nodeKey(a,l);if(!this.containerNodes.has(u))continue;let c=t[u];c==null&&(c=0);const h=this.inputLayersTensorIndices[o];r.push([a.name,c,h])}e.inputLayers=r;const i=[];for(let o=0;o<this.outputLayers.length;o++){const a=this.outputLayers[o],l=this.outputLayersNodeIndices[o],u=en.nodeKey(a,l);if(!this.containerNodes.has(u))continue;let c=t[u];c==null&&(c=0);const h=this.outputLayersTensorIndices[o];i.push([a.name,c,h])}return e.outputLayers=i,e}static fromConfig(e,t,n={},r=!1){const i={},o={};function a(m,S){m.name in o?o[m.name].push(S):o[m.name]=[S]}function l(m,S){const b=[];let f;for(const _ of S){const v=_[0],T=_[1],N=_[2];if(f=_[3]==null?{}:_[3],!(v in i)){a(m,S);return}const O=i[v];if(O.inboundNodes.length<=T){a(m,S);return}const $=O.inboundNodes[T];b.push($.outputTensors[N])}b.length>0&&m.apply(_t(b),f)}function u(m){const S=m.name,b=Od(m,t.customObjects!=null?t.customObjects:{});b.setFastWeightInitDuringBuild(r),i[S]=b,m.inboundNodes.forEach(_=>{if(!(_ instanceof Array))throw new W(`Corrupted configuration, expected array for nodeData: ${_}`);a(b,_)})}const c=t.name,h=t.layers;for(const m of h)u(m);for(;!Bx(o);)for(const m of h){const S=i[m.name];if(S.name in o){const b=o[S.name];delete o[S.name];for(const f of b)l(S,f)}}const d=[],w=[],I=t.inputLayers;for(const m of I){const S=m[0],b=m[1],f=m[2];mn(S in i);const v=i[S].inboundNodes[b].outputTensors;d.push(v[f])}const E=t.outputLayers;for(const m of E){const S=m[0],b=m[1],f=m[2];mn(S in i);const v=i[S].inboundNodes[b].outputTensors;w.push(v[f])}return new e({inputs:d,outputs:w,name:c})}get stateful(){if(this._stateful)throw new W("Container instance unexpectedly has _stateful = true. The statefulness of a Container is determined by the Layers it contains. Its _stateful property must remain the default false.");for(const e of this.layers)if(e.stateful)return!0;return!1}resetStates(){Y(()=>{this.layers.forEach(e=>{e.stateful&&e.resetStates()})})}}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function nv(s,e,t){const n=e.length;if(s==null||Array.isArray(s)&&s.length===0)return e.map(r=>null);if(n===1)return Array.isArray(s)&&s.length===1?s:typeof s=="object"&&e[0]in s?[s[e[0]]]:[s];if(Array.isArray(s)){if(s.length!==n)throw new Error(`Provided ${t} is an array of ${s.length} element(s), but the model has ${n} outputs. Make sure a set of weights is provided for each model output.`);return s}else if(typeof s=="object"&&Object.keys(s).length>0&&typeof s[Object.keys(s)[0]]=="object"){const r=[];return e.forEach(i=>{i in s?r.push(s[i]):r.push(null)}),r}else throw new Error(`The model has multiple (${n}) outputs, so ${t} must be either an array with ${n} elements or an object with ${e} keys. Provided ${t} not understood: ${JSON.stringify(s)}`)}function zd(s,e){return nv(s,e,"classWeight")}async function Vd(s,e,t,n){if(t!=null){const r=Y(()=>{if(s.shape.length===1)return Jn(s);if(s.shape.length===2){if(s.shape[1]>1)return $i(s,1);if(s.shape[1]===1)return se(s,[s.shape[0]]);throw new Error(`Encountered unexpected last-dimension size (${s.shape[1]}) during handling of class weights. The size is expected to be >= 1.`)}else throw new Error(`Unexpected rank of target (y) tensor (${s.rank}) during handling of class weights. The rank is expected to be 1 or 2.`)}),i=Array.from(await r.data());Ce(r);const o=[];return i.forEach(a=>{if(t[a]==null)throw new Error(`classWeight must contain all classes in the training data. The class ${a} exists in the data but not in classWeight`);o.push(t[a])}),pt(o,"float32")}else return null}function sv(s,e){return J(s,e)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */const rv=32;function Gd(s,e){let t,n;const r=e;t=r.xs,n=r.ys,P(t!=null&&n!=null,()=>`A Dataset iterator for fitDataset() is expected to generate objects of the form \`{xs: xVal, ys: yVal}\`, where the two values may be \`tf.Tensor\`, an array of Tensors, or a map of string to Tensor.  The provided Dataset instead generates ${e}`);const i=Yu("input",s.inputNames,t),o=Yu("output",s.outputNames,n),a=i[0].shape[0];P(i.length===s.inputs.length,()=>`LayersModel has ${s.inputs.length} inputs, but the dataset provides ${i.length} inputs.  (Expected input keys: ${JSON.stringify(s.inputNames)})`),P(o.length===s.outputs.length,()=>`LayersModel has ${s.outputs.length} outputs, but the dataset provides ${o.length} outputs.  (Expected output keys: ${JSON.stringify(s.outputNames)})`);for(let l=0;l<i.length;l++)P(i[l].shape[0]===a,()=>`Batch size mismatch: input ${s.inputNames[l]} has ${i[l].shape[0]}; expected  ${a} based on input ${s.inputNames[0]}.`);for(let l=0;l<o.length;l++)P(o[l].shape[0]===a,()=>`Batch size mismatch: output ${s.outputNames[l]} has ${o[l].shape[0]}; expected  ${a} based on input ${s.inputNames[0]}.`);return{xs:i,ys:o}}function Yu(s,e,t){if(t instanceof et)return[t];if(Array.isArray(t))return P(t.length===e.length,()=>`Received an array of ${t.length} Tensors, but expected ${e.length} to match the ${s} keys ${e}.`),t;{const n=[];for(const r of e){if(t[r]==null)throw new W(`The feature data generated by the dataset lacks the required ${s} key '${r}'.`);n.push(t[r])}return n}}function iv(s){if(s.length===3)throw new be("Validation with sample weights is not implemented yet.");return{xs:s[0],ys:s[1]}}async function ov(s,e,t){const n=t.batchesPerEpoch!=null;if(P(s.optimizer!=null,()=>"You must compile a model before training/testing. Use LayersModel.compile(modelCompileConfig)."),P(t!=null,()=>"For fitDataset(), the 2nd argument (config) is required, but it is not provided in this call."),P(t.epochs!=null&&t.epochs>0&&Number.isInteger(t.epochs),()=>`For fitDataset(), config.epochs is expected to be a positive integer, but got ${t.epochs}`),P(!n||t.batchesPerEpoch>0&&Number.isInteger(t.batchesPerEpoch),()=>`For fitDataset(), config.batchesPerEpoch is expected to be a positive integer if specified, but got ${t.batchesPerEpoch}`),P(t.validationSplit==null,()=>"`validationSplit` is not supported by `fitDataset()`. Use validationData instead."),s.isTraining)throw new Error("Cannot start training because another fit() call is ongoing.");s.isTraining=!0;try{const r=t.validationData!=null;let i,o;if(r)if(Qu(t.validationData))P(t.validationBatches==null||t.validationBatches>0&&Number.isInteger(t.validationBatches),()=>`For fitDataset() with dataset-based validation, config.validationBatches is expected not to be provided, or to be a positive integer, but got ${t.validationBatches}`);else{const m=iv(t.validationData);i=m.xs,o=m.ys}const a=s.makeTrainFunction(),l=s.getDedupedMetricsNames();let u;r?u=l.slice().concat(l.map(m=>"val_"+m)):u=l.slice();const c=Nd(t.callbacks,t.yieldEvery),h=t.verbose==null?1:t.verbose,{callbackList:d,history:w}=Dd(c,h,t.epochs,null,null,av(e,t),null,r,u);d.setModel(s),s.history=w,await d.onTrainBegin(),s.stopTraining_=!1;let I=t.initialEpoch==null?0:t.initialEpoch,E=await e.iterator();for(;I<t.epochs;){const m={};await d.onEpochBegin(I);let S=0,b=0;for(n||(E=await e.iterator());!n||S<t.batchesPerEpoch;){const f=await E.next();if(n&&f.done){console.warn(`You provided \`batchesPerEpoch\` as ${t.batchesPerEpoch}, but your dataset iterator ran out of data after ${S} batches; interrupting training. Make sure that your dataset can generate at least \`batchesPerEpoch * epochs\` batches (in this case, ${t.batchesPerEpoch*t.epochs} batches). You may need to use the repeat() function when building your dataset.`);break}if(f.value!=null){const{xs:_,ys:v}=Gd(s,f.value),T={};T.batch=b,T.size=_[0].shape[0],await d.onBatchBegin(b,T);const N=[];if(t.classWeight!=null){const A=zd(t.classWeight,s.outputNames);for(let g=0;g<A.length;++g)N.push(await Vd(v[g],null,A[g]))}const O=_.concat(v).concat(N),$=a(O);Ce(O);for(let A=0;A<l.length;++A){const g=l[A],p=$[A];T[g]=p,As(p)}await d.onBatchEnd(b,T),$d(T),b++,S++}if(n?S>=t.batchesPerEpoch:f.done){if(r){let _;Qu(t.validationData)?_=ke(await s.evaluateDataset(t.validationData,{batches:t.validationBatches})):_=ke(s.evaluate(i,o,{batchSize:t.validationBatchSize==null?rv:t.validationBatchSize,verbose:0}));for(let v=0;v<s.metricsNames.length;++v)m[`val_${s.metricsNames[v]}`]=_[v]}break}if(s.stopTraining_)break}if(await d.onEpochEnd(I,m),I++,s.stopTraining_)break}return await d.onTrainEnd(),await s.history.syncData(),s.history}finally{s.isTraining=!1}}function av(s,e){let t=null;return e.batchesPerEpoch!=null?t=e.batchesPerEpoch:Number.isFinite(s.size)&&(t=s.size),t}function Qu(s){return typeof s.iterator=="function"}function lv(s){return typeof s.next=="function"}async function uv(s,e,t){t=t||{};const n=t.batches!=null,r=s.testFunction;let i=[];if(t.verbose>0)throw new be("Verbose mode is not implemented yet.");P(!n||t.batches>0&&Number.isInteger(t.batches),()=>`Test loop expects \`batches\` to be a positive integer, but received ${JSON.stringify(t.batches)}`);const o=lv(e)?e:await e.iterator();let a=0,l=0;for(;!n||l<t.batches;){const u=await o.next();if(i=Y(()=>{if(u.value){const{xs:c,ys:h}=Gd(s,u.value),d=c.concat(h),w=Y(()=>r(d));if(Ce(d),l===0)for(let E=0;E<w.length;++E)i.push(Rt(0));const I=d[0].shape[0];for(let E=0;E<w.length;++E){const m=w[E],S=i[E];i[E]=Y(()=>ae(i[E],J(I,m))),l>0&&Ce(S)}Ce(w),a+=I,++l}return i}),u.done){n&&console.warn(`Your dataset iterator ran out of data during evaluateDataset(). Interrupting evalution. Make sure that your dataset can generate at least \`batches\` batches (in this case, ${t.batches} batches). You may need to use the repeat() function when building your dataset.`);break}}for(let u=0;u<i.length;++u){const c=i[u];i[u]=ge(i[u],a),Ce(c)}return _t(i)}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function jo(s){P(s>0&&Number.isInteger(s),()=>`batchSize is required to be a positive integer, but got ${s}`)}function er(s,e,t){return s==null?[null]:Array.isArray(s)?s.map(n=>ss(n,e,t-e)):ss(s,e,t-e)}function Ta(s,e){return Y(()=>s==null?null:Array.isArray(s)?s.map(t=>Ta(t,e)):jx(s,e.dtype==="int32"?e:Ee(e,"int32")))}function Ko(s,e){const t=[];let n=0,r=null;for(;n<s;)r=n+e,r>=s&&(r=s),t.push([n,r]),n=r;return t}function Wd(s){const e=[];s instanceof et&&(s=[s]);for(let t=0;t<s.length;++t){const n=s[t];if(n.rank===1)e.push(al(n,1));else{if(n.rank===0)throw new Error("Expected tensor to be at least 1D, but received a 0D tensor (scalar).");e.push(n)}}return e}function Yt(s,e){if(s==null)return;const t=[];if(e instanceof et)t.push(e.id);else if(Array.isArray(e))e.forEach(r=>t.push(r.id));else if(e!=null)for(const r in e){const i=e[r];t.push(i.id)}const n=[];if(s instanceof et)t.indexOf(s.id)===-1&&n.push(s);else if(Array.isArray(s))s.forEach(r=>{t.indexOf(r.id)===-1&&n.push(r)});else if(s!=null)for(const r in s){const i=s[r];t.indexOf(i.id)===-1&&n.push(i)}n.forEach(r=>{r.isDisposed||r.dispose()})}/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */function cv(s){return s instanceof et}function Ea(s){return Array.isArray(s)}function Zu(s){return!cv(s)&&!Ea(s)}function Ju(s,e,t,n=!0,r=""){if(e==null||e.length===0){if(s!=null){let o=!1;if(Ea(s)&&s.length>0)o=!0;else if(Zu(s)){for(const a in s)if(s.hasOwnProperty(a)){o=!0;break}}else o=!0;if(o)throw new W(`Error when checking model ${r} expected no data, but got ${s}`)}return[]}if(s==null)return e.map(o=>null);let i;if(Zu(s)){s=s,i=[];for(const o of e){if(s[o]==null)throw new W(`No data provided for "${o}". Need data for each key in: ${e}`);i.push(s[o])}}else if(Ea(s)){if(s=s,s.length!==e.length)throw new W(`Error when checking model ${r}: the Array of Tensors that you are passing to your model is not the size the model expected. Expected to see ${e.length} Tensor(s), but instead got the following list of Tensor(s): ${s}`);i=s}else{if(s=s,e.length>1)throw new W(`The model ${r} expects ${e.length} Tensor(s), but only received one Tensor. Found: Tensor with shape ${s.shape}`);i=[s]}if(i=Wd(i),t!=null)for(let o=0;o<e.length;++o){if(t[o]==null)continue;const a=i[o];if(a.shape.length!==t[o].length)throw new W(`Error when checking ${r}: expected ${e[o]} to have ${t[o].length} dimension(s). but got array with shape ${a.shape}`);for(let l=0;l<t[o].length;++l){if(l===0&&!n)continue;const u=a.shape[l],c=t[o][l];if(c!=null&&c>=0&&u!==c)throw new W(`${r} expected a batch of elements where each example has shape [${t[o].slice(1,t[o].length)}] (i.e.,tensor shape [*,${t[o].slice(1,t[o].length)}]) but the ${r} received an input with ${a.shape[0]} examples, each with shape [${a.shape.slice(1,a.shape.length)}] (tensor shape [${a.shape}])`)}}return i}function hv(s,e,t){const n=ns(s.map(i=>i.shape[0]));n.sort();const r=ns(e.map(i=>i.shape[0]));if(r.sort(),n.length>1)throw new W(`All input Tensors (x) should have the same number of samples. Got array shapes: ${JSON.stringify(s.map(i=>i.shape))}`);if(r.length>1)throw new W(`All target Tensors (y) should have the same number of samples. Got array shapes: ${JSON.stringify(e.map(i=>i.shape))}`);if(n.length>0&&r.length>0&&!Ht(n,r))throw new W(`Input Tensors should have the same number of samples as target Tensors. Found ${n[0]} input sample(s) and ${r[0]} target sample(s).`)}function fv(s,e,t){const n=[xo,vo,Sr];for(let r=0;r<s.length;++r){const i=s[r],o=e[r],a=t[r];if(o!=null){if(o===Sr&&i.shape[i.shape.length-1]===1)throw new W(`You are passing a target array of shape ${i.shape} while using a loss 'categorical_crossentropy'. 'categorical_crossentropy'expects targets to be binary matrices (1s and 0s) of shape [samples, classes].`);if(n.indexOf(o)!==-1){const l=i.shape.slice(1),u=a.slice(1);for(let c=0;c<l.length;++c){const h=l[c],d=u[c];if(d!=null&&h!==d)throw new W(`A target Tensor with shape ${i.shape} was passed for an output of shape ${a}, while using a loss function that expects targets to have the same shape as the output.`)}}}}}function ec(s,e,t,n=!0,r=""){let i;if(Array.isArray(s)){if(s.length!==e.length)throw new W(`Error when checking model ${r}: the Array of Tensors that you are passing to your model is not the size the the model expected. Expected to see ${e.length} Tensor(s), but instead got ${s.length} Tensors(s).`);i=s}else{if(e.length>1)throw new W(`The model expects ${e.length} ${r} Tensors, but only received one Tensor. Found: array with shape ${JSON.stringify(s.shape)}.`);i=[s]}if(t!=null)for(let o=0;o<e.length;++o){if(t[o]==null)continue;const a=i[o];if(a.shape.length!==t[o].length)throw new W(`Error when checking ${r}: expected ${e[o]} to have ${t[o].length} dimension(s), but got array with shape ${JSON.stringify(a.shape)}`);for(let l=0;l<t[o].length;++l){if(l===0&&!n)continue;const u=a.shape[l],c=t[o][l];if(c!=null&&c!==u)throw new W(`Error when checking ${r}: expected ${e[o]} to have shape ${JSON.stringify(t[o])} but got array with shape ${JSON.stringify(a.shape)}.`)}}}function dv(s,e){if(s==null||Array.isArray(s)&&s.length===0)return e.map(n=>[]);let t;if(typeof s=="string"||typeof s=="function")t=[s];else if(Array.isArray(s)||typeof s=="object")t=s;else throw new TypeError(`Type of metrics argument not understood. Expected an string,function, Array, or Object, found: ${s}`);if(Array.isArray(t))return e.map(n=>t);{const n=[];for(const r of e){let i=t.hasOwnProperty(r)?t[r]:[];Array.isArray(i)||(i=[i]),n.push(i)}return n}}const pv="layers-model";class _o extends en{constructor(e){super(e),this.isTraining=!1}summary(e,t,n=console.log){if(!this.built)throw new W("This model has never been called, thus its weights have not been created yet. So no summary can be displayed. Build the model first (e.g., by calling it on some test data).");q2(this,e,t,n)}compile(e){if(e.loss==null&&(e.loss=[]),this.loss=e.loss,typeof e.optimizer=="string")this.optimizer_=W2(e.optimizer),this.isOptimizerOwned=!0;else{if(!(e.optimizer instanceof qn))throw new W("User-defined optimizer must be an instance of tf.Optimizer.");this.optimizer_=e.optimizer,this.isOptimizerOwned=!1}let t=[];if(!Array.isArray(e.loss)&&typeof e.loss!="string"&&typeof e.loss!="function"){e.loss=e.loss;for(const o in e.loss)if(this.outputNames.indexOf(o)===-1)throw new W(`Unknown entry in loss dictionary: "${o}". Only expected the following keys: ${this.outputNames}`);for(const o of this.outputNames)e.loss[o]==null&&console.warn(`Output "${o}" is missing from loss dictionary. We assume this was done on purpose, and we will not be expecting data to be passed to ${o} during training`),t.push(Ho(e.loss[o]))}else if(Array.isArray(e.loss)){if(e.loss.length!==this.outputs.length)throw new W(`When passing an Array as loss, it should have one entry per model output. The model has ${this.outputs.length} output(s), but you passed loss=${e.loss}.`);t=e.loss.map(a=>Ho(a))}else{const o=Ho(e.loss);this.outputs.forEach(a=>{t.push(o)})}this.lossFunctions=t,this.feedOutputNames=[],this.feedOutputShapes=[],this.feedLossFns=[];for(let o=0;o<this.outputs.length;++o){const a=this.internalOutputShapes[o],l=this.outputNames[o];this.feedOutputNames.push(l),this.feedOutputShapes.push(a),this.feedLossFns.push(this.lossFunctions[o])}const n=[];this.metrics=e.metrics,this.metricsNames=["loss"],this.metricsTensors=[],_i("loss",()=>{for(let o=0;o<this.outputs.length;++o){if(n.indexOf(o)!==-1)continue;const a=this.lossFunctions[o];this.outputs.length>1&&(this.metricsTensors.push([a,o]),this.metricsNames.push(this.outputNames[o]+"_loss"))}});const r=dv(e.metrics,this.outputNames),i=(o,a,l)=>{this.outputNames.length>1&&(a=this.outputNames[o]+"_"+a),this.metricsNames.push(a),this.metricsTensors.push([l,o])};_i("metric",()=>{for(let o=0;o<this.outputs.length;++o){if(n.indexOf(o)!==-1)continue;const a=r[o];(u=>{let h,d,w;for(const I of u){if(typeof I=="string"&&["accuracy","acc","crossentropy","ce"].indexOf(I)!==-1){const m=this.internalOutputShapes[o];m[m.length-1]===1||this.lossFunctions[o]===vo?["accuracy","acc"].indexOf(I)!==-1?d=Md:["crossentropy","ce"].indexOf(I)!==-1&&(d=M2):this.lossFunctions[o]===qi?["accuracy","acc"].indexOf(I)!==-1?d=P2:["crossentropy","ce"].indexOf(I)!==-1&&(d=Ld):["accuracy","acc"].indexOf(I)!==-1?d=Pd:["crossentropy","ce"].indexOf(I)!==-1&&(d=Rd);let S;["accuracy","acc"].indexOf(I)!==-1?S="acc":["crossentropy","ce"].indexOf(I)!==-1&&(S="ce"),w=d,h=""+S}else w=G2(I),h=""+Jr(I);let E;_i(h,()=>{E=w}),i(o,h,E)}})(a)}}),this.collectedTrainableWeights=this.trainableWeights}checkTrainableWeightsConsistency(){this.collectedTrainableWeights!=null&&this.trainableWeights.length!==this.collectedTrainableWeights.length&&console.warn("Discrepancy between trainableweights and collected trainable weights. Did you set `model.trainable` without calling `model.compile()` afterwards?")}evaluate(e,t,n={}){const r=n.batchSize==null?32:n.batchSize;jo(r);const o=this.standardizeUserDataXY(e,t,!0,r);try{const a=o[0].concat(o[1]);this.makeTestFunction();const l=this.testFunction,u=this.testLoop(l,a,r,n.verbose,n.steps);return _t(u)}finally{Yt(o[0],e),Yt(o[1],t)}}async evaluateDataset(e,t){return this.makeTestFunction(),uv(this,e,t)}checkNumSamples(e,t,n,r="steps"){let i;if(n!=null){if(i=null,t!=null)throw new W(`If ${r} is set, batchSize must be null or undefined.Got batchSize = ${t}`)}else if(e!=null)Array.isArray(e)?i=e[0].shape[0]:i=e.shape[0];else throw new W(`Either the input data should have a defined shape, or ${r} shoud be specified.`);return i}execute(e,t){if(Array.isArray(t)&&t.length===0)throw new W("`outputs` is an empty Array, which is not allowed.");const n=Array.isArray(t),r=n?t:[t],i=this.retrieveSymbolicTensors(r),o=new Rn;if(e instanceof et&&(e=[e]),Array.isArray(e)){if(e.length!==this.inputs.length)throw new W(`The number of inputs provided (${e.length}) does not match the number of inputs of this model (${this.inputs.length}).`);for(let l=0;l<this.inputs.length;++l)o.add(this.inputs[l],e[l])}else for(const l of this.inputs){const u=e[l.name];if(u==null)throw new W(`No value is provided for the model's input ${l.name}`);o.add(l,u)}const a=rr(i,o);return n?a:a[0]}retrieveSymbolicTensors(e){const t=Li(null,e.length);let n=e.length;for(const r of this.layers){const i=Array.isArray(r.output)?r.output:[r.output],o=i.map(a=>a.name);for(let a=0;a<e.length;++a){const l=o.indexOf(e[a]);if(l!==-1&&(t[a]=i[l],n--),n===0)break}if(n===0)break}if(n>0){const r=[];throw t.forEach((i,o)=>{i==null&&r.push(e[o])}),new W(`Cannot find SymbolicTensors for output name(s): ${JSON.stringify(r)}`)}return t}predictLoop(e,t=32,n=!1){return Y(()=>{const r=this.checkNumSamples(e);if(n)throw new be("Verbose predictLoop() is not implemented yet.");const i=Ko(r,t),o=this.outputs.map(a=>[]);for(let a=0;a<i.length;++a)Y(()=>{const u=i[a][0],c=i[a][1],h=er(e,u,c),d=[];if(Array.isArray(h))for(let I=0;I<h.length;++I)d.push({key:this.inputs[I],value:h[I]});else d.push({key:this.inputs[0],value:h});const w=new Rn(d);return rr(this.outputs,w)}).forEach((u,c)=>o[c].push(u));return _t(o.map(a=>es(a,0)))})}predict(e,t={}){const n=Wd(e);ec(n,this.inputNames,this.feedInputShapes,!1);try{const r=t.batchSize==null?32:t.batchSize;return jo(r),this.predictLoop(n,r)}finally{Yt(n,e)}}predictOnBatch(e){ec(e,this.inputNames,this.feedInputShapes,!0);const t=(Array.isArray(e)?e[0]:e).shape[0];return this.predictLoop(e,t)}standardizeUserDataXY(e,t,n=!0,r){if(this.optimizer_==null)throw new zn("You must compile a model before training/testing. Use LayersModel.compile(modelCompileArgs).");const i=[];for(let o=0;o<this.feedOutputShapes.length;++o){const a=this.feedOutputShapes[o];this.feedLossFns[o]===qi?i.push(a.slice(0,a.length-1).concat([1])):i.push(a)}if(e=Ju(e,this.feedInputNames,this.feedInputShapes,!1,"input"),t=Ju(t,this.feedOutputNames,i,!1,"target"),hv(e,t),fv(t,this.feedLossFns,this.feedOutputShapes),this.stateful&&r!=null&&r>0&&e[0].shape[0]%r!==0)throw new W(`In a stateful network, you should only pass inputs with a number of samples that is divisible by the batch size ${r}. Found: ${e[0].shape[0]} sample(s).`);return[e,t]}async standardizeUserData(e,t,n,r,i=!0,o){const[a,l]=this.standardizeUserDataXY(e,t,i,o);if(n!=null)throw new Error("sample weight is not supported yet.");let u=null;if(r!=null){const c=zd(r,this.outputNames);u=[];for(let h=0;h<c.length;++h)u.push(await Vd(l[h],null,c[h]))}return[a,l,u]}testLoop(e,t,n,r=0,i){return Y(()=>{const o=this.checkNumSamples(t,n,i,"steps"),a=[];if(r>0)throw new be("Verbose mode is not implemented yet.");if(i!=null)throw new be("steps mode in testLoop() is not implemented yet");{const l=Ko(o,n),u=pt(Bi(0,o));for(let c=0;c<l.length;++c){const h=l[c][0],d=l[c][1],w=ss(u,h,d-h),I=Ta(t,w),E=e(I);if(c===0)for(let m=0;m<E.length;++m)a.push(Rt(0));for(let m=0;m<E.length;++m){const S=E[m];a[m]=ae(a[m],J(d-h,S))}}for(let c=0;c<a.length;++c)a[c]=ge(a[c],o)}return a})}getDedupedMetricsNames(){const e=this.metricsNames,t=[];for(let n=0;n<e.length;++n){const r=e[n];let i=r;if(Ou(e,r)>1){const o=Ou(e.slice(0,n),r);i+=`_${o}`}t.push(i)}return t}makeTrainFunction(){return e=>{const t=[],n=e.slice(0,this.inputs.length),r=e.slice(this.inputs.length,this.inputs.length+this.outputs.length),i=e.slice(this.inputs.length+this.outputs.length,this.inputs.length+this.outputs.length*2),o=[],a=()=>{const h=[];for(let E=0;E<this.inputs.length;++E)h.push({key:this.inputs[E],value:n[E]});const d=new Rn(h),w=rr(this.outputs,d,{training:!0});let I;for(let E=0;E<this.lossFunctions.length;++E){const m=this.lossFunctions[E];let S=m(r[E],w[E]);i[E]!=null&&(S=sv(S,i[E]));const b=Xe(S);t.push(b),E===0?I=S:I=ae(I,S)}for(let E=0;E<this.metricsTensors.length;++E){let m;if(this.outputs.length>1&&E<this.outputs.length)m=t[E];else{const S=this.metricsTensors[E][0],b=this.metricsTensors[E][1];m=Xe(S(r[b],w[b]))}As(m),o.push(m)}return I=Xe(I),this.calculateLosses().forEach(E=>{I=ae(I,E)}),I},l=this.collectedTrainableWeights.map(h=>h.read());return[this.optimizer_.minimize(a,!0,l)].concat(o)}}makeTestFunction(){this.testFunction=e=>Y(()=>{const t=[];let n;const r=e.slice(0,this.inputs.length),i=e.slice(this.inputs.length,this.inputs.length+this.outputs.length),o=[];for(let u=0;u<this.inputs.length;++u)o.push({key:this.inputs[u],value:r[u]});const a=new Rn(o),l=rr(this.outputs,a);for(let u=0;u<this.lossFunctions.length;++u){const c=this.lossFunctions[u],h=Xe(c(i[u],l[u]));u===0?n=h:n=ae(n,h),t.push(n)}for(let u=0;u<this.metricsTensors.length;++u){const c=this.metricsTensors[u][0],h=this.metricsTensors[u][1],d=Xe(c(i[h],l[h]));t.push(d)}return t})}async fit(e,t,n={}){if(this.isTraining)throw new Error("Cannot start training because another fit() call is ongoing.");this.isTraining=!0;let r,i,o,a,l,u,c,h,d;try{const w=n.batchSize==null?32:n.batchSize;jo(w);const E=await this.standardizeUserData(e,t,n.sampleWeight,n.classWeight,!1,w);r=E[0],i=E[1],d=E[2];let m=!1,S;if(n.validationData!=null&&n.validationData.length>0){if(m=!0,n.validationData.length===2)l=n.validationData[0],u=n.validationData[1];else throw n.validationData.length===3?new be("validationData including sample weights is not supported yet."):new W(`When passing validation data, it must contain 2 (valX, valY) or 3 (valX, valY, valSampleWeight) items; ${n.validationData} is invalid.`);const A=await this.standardizeUserData(l,u,null,null,!0,w);c=A[0],h=A[1],S=c.concat(h)}else if(n.validationSplit!=null&&n.validationSplit>0&&n.validationSplit<1){m=!0;const $=Math.floor(r[0].shape[0]*(1-n.validationSplit)),A=r[0].shape[0];c=er(r,$,A),o=r,r=er(r,0,$),h=er(i,$,A),a=i,i=er(i,0,$),S=c.concat(h)}else n.validationSteps!=null&&(m=!0);const b=r.concat(i).concat(d);this.checkTrainableWeightsConsistency();const f=this.makeTrainFunction(),_=this.getDedupedMetricsNames();let v,T;m?(this.makeTestFunction(),v=this.testFunction,T=_.slice().concat(_.map($=>"val_"+$))):(v=null,S=[],T=_.slice());const N=Nd(n.callbacks,n.yieldEvery);return await this.fitLoop(f,b,_,w,n.epochs,n.verbose,N,v,S,n.shuffle,T,n.initialEpoch,null,null)}finally{this.isTraining=!1,Yt(r,e),Yt(i,t),Yt(o,e),Yt(a,t),Yt(c,l),Yt(h,u),d!=null&&Ce(d)}}async fitLoop(e,t,n,r,i,o,a,l,u,c,h,d,w,I){r==null&&(r=32),i==null&&(i=1),c==null&&(c=!0),d==null&&(d=0);let E=!1;if(l!=null&&u!=null&&(E=!0),I!=null&&(E=!0,w==null))throw new W("Can only use `validationSteps` when doing step-wise training, i.e., `stepsPerEpoch` must be set.");const m=this.checkNumSamples(t,r,w,"steps_per_epoch");let S;m!=null&&(S=Bi(0,m)),o==null&&(o=1);const{callbackList:b,history:f}=Dd(a,o,i,d,m,w,r,E,h);b.setModel(this),this.history=f,await b.onTrainBegin(),this.stopTraining_=!1;for(let _=d;_<i;++_){await b.onEpochBegin(_);const v={};if(w!=null)throw new be("stepsPerEpoch mode is not implemented yet.");{if(c==="batch")throw new be("batch shuffling is not implemneted yet");c&&wm(S);const T=pt(S),N=Ko(m,r);for(let O=0;O<N.length;++O){const $={};if(await b.onBatchBegin(O,$),Y(()=>{const A=N[O][0],g=N[O][1],p=ss(T,A,g-A);$.batch=O,$.size=g-A;const y=Ta(t,p),x=e(y);for(let k=0;k<n.length;++k){const C=n[k],R=x[k];$[C]=R,As(R)}if(O===N.length-1&&E){const k=this.testLoop(l,u,r);for(let C=0;C<n.length;++C){const R=n[C],z=k[C];As(z),v["val_"+R]=z}}}),await b.onBatchEnd(O,$),$d($),this.stopTraining_)break}T.dispose()}if(await b.onEpochEnd(_,v),this.stopTraining_)break}return await b.onTrainEnd(),await this.history.syncData(),this.history}async fitDataset(e,t){return ov(this,e,t)}async trainOnBatch(e,t){const n=await this.standardizeUserData(e,t),r=n[0],i=n[1],a=this.makeTrainFunction()(r.concat(i)),l=[];for(const u of a){const c=await u.data();l.push(c[0])}return Ce(a),Yt(n[0],e),Yt(n[1],t),_t(l)}getNamedWeights(e){const t=[],n=e!=null&&e.trainableOnly,r=n?this.trainableWeights:this.weights,i=this.getWeights(n);for(let o=0;o<r.length;++o)n&&!r[o].trainable||t.push({name:r[o].originalName,tensor:i[o]});return t}set stopTraining(e){this.stopTraining_=e}get stopTraining(){return this.stopTraining_}get optimizer(){return this.optimizer_}set optimizer(e){this.optimizer_!==e&&(this.optimizer_=e,this.isOptimizerOwned=!1)}dispose(){const e=super.dispose();if(e.refCountAfterDispose===0&&this.optimizer!=null&&this.isOptimizerOwned){const t=pu().numTensors;this.optimizer_.dispose(),e.numDisposedVariables+=t-pu().numTensors}return e}getLossIdentifiers(){let e;if(typeof this.loss=="string")e=In(this.loss);else if(Array.isArray(this.loss)){for(const t of this.loss)if(typeof t!="string")throw new Error("Serialization of non-string loss is not supported.");e=this.loss.map(t=>In(t))}else{const t=Object.keys(this.loss);e={};const n=this.loss;for(const r of t)if(typeof n[r]=="string")e[r]=In(n[r]);else throw new Error("Serialization of non-string loss is not supported.")}return e}getMetricIdentifiers(){if(typeof this.metrics=="string"||typeof this.metrics=="function")return[In(Jr(this.metrics))];if(Array.isArray(this.metrics))return this.metrics.map(e=>In(Jr(e)));{const e={};for(const t in this.metrics)e[t]=In(Jr(this.metrics[t]));return e}}getTrainingConfig(){return{loss:this.getLossIdentifiers(),metrics:this.getMetricIdentifiers(),optimizer_config:{class_name:this.optimizer.getClassName(),config:this.optimizer.getConfig()}}}loadTrainingConfig(e){if(e.weighted_metrics!=null)throw new Error("Loading weight_metrics is not supported yet.");if(e.loss_weights!=null)throw new Error("Loading loss_weights is not supported yet.");if(e.sample_weight_mode!=null)throw new Error("Loading sample_weight_mode is not supported yet.");const t=Ia(e.optimizer_config),n=Od(t);let r;if(typeof e.loss=="string")r=Yn(e.loss);else if(Array.isArray(e.loss))r=e.loss.map(o=>Yn(o));else if(e.loss!=null){r={};for(const o in e.loss)r[o]=Yn(e.loss[o])}let i;if(Array.isArray(e.metrics))i=e.metrics.map(o=>Yn(o));else if(e.metrics!=null){i={};for(const o in e.metrics)i[o]=Yn(e.metrics[o])}this.compile({loss:r,metrics:i,optimizer:n})}async save(e,t){if(typeof e=="string"){const u=Yy(e);if(u.length===0)throw new W(`Cannot find any save handlers for URL '${e}'`);if(u.length>1)throw new W(`Found more than one (${u.length}) save handlers for URL '${e}'`);e=u[0]}if(e.save==null)throw new W("LayersModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");const n=await gu(this.getNamedWeights(t)),a={modelTopology:this.toJSON(null,!1),format:pv,generatedBy:`TensorFlow.js tfjs-layers v${Fd}`,convertedBy:null};if((t==null?!1:t.includeOptimizer)&&this.optimizer!=null){a.trainingConfig=this.getTrainingConfig();const u="optimizer",{data:c,specs:h}=await gu(await this.optimizer.getWeights(),u);n.specs.push(...h),n.data=Xy([n.data,c])}return this.userDefinedMetadata!=null&&(Hu(this.userDefinedMetadata,this.name,!0),a.userDefinedMetadata=this.userDefinedMetadata),a.weightData=n.data,a.weightSpecs=n.specs,e.save(a)}setUserDefinedMetadata(e){Hu(e,this.name),this.userDefinedMetadata=e}getUserDefinedMetadata(){return this.userDefinedMetadata}}_o.className="Model";re(_o);class qd extends _o{}qd.className="Functional";re(qd);const mv="This is not an object",gv="This is not a Float16Array object",tc="This constructor is not a subclass of Float16Array",Hd="The constructor property value is not an object",yv="Species constructor didn't return TypedArray object",bv="Derived constructor created TypedArray object which was too small length",fr="Attempting to access detached ArrayBuffer",Aa="Cannot convert undefined or null to object",Ca="Cannot mix BigInt and other types, use explicit conversions",nc="@@iterator property is not callable",sc="Reduce of empty array with no initial value",wv="The comparison function must be either a function or undefined",Xo="Offset is out of bounds";function $e(s){return(e,...t)=>vt(s,e,t)}function Ks(s,e){return $e(Ms(s,e).get)}const{apply:vt,construct:ir,defineProperty:rc,get:Yo,getOwnPropertyDescriptor:Ms,getPrototypeOf:Br,has:$a,ownKeys:jd,set:ic,setPrototypeOf:Kd}=Reflect,xv=Proxy,{EPSILON:vv,MAX_SAFE_INTEGER:oc,isFinite:Xd,isNaN:Ps}=Number,{iterator:xn,species:_v,toStringTag:Sl,for:Sv}=Symbol,Rs=Object,{create:So,defineProperty:Fr,freeze:Iv,is:ac}=Rs,Na=Rs.prototype,kv=Na.__lookupGetter__?$e(Na.__lookupGetter__):(s,e)=>{if(s==null)throw De(Aa);let t=Rs(s);do{const n=Ms(t,e);if(n!==void 0)return An(n,"get")?n.get:void 0}while((t=Br(t))!==null)},An=Rs.hasOwn||$e(Na.hasOwnProperty),Yd=Array,Qd=Yd.isArray,Io=Yd.prototype,Tv=$e(Io.join),Ev=$e(Io.push),Av=$e(Io.toLocaleString),Il=Io[xn],Cv=$e(Il),{abs:$v,trunc:Zd}=Math,ko=ArrayBuffer,Nv=ko.isView,Jd=ko.prototype,Dv=$e(Jd.slice),Ov=Ks(Jd,"byteLength"),Da=typeof SharedArrayBuffer<"u"?SharedArrayBuffer:null,Mv=Da&&Ks(Da.prototype,"byteLength"),kl=Br(Uint8Array),Pv=kl.from,Qe=kl.prototype,Rv=Qe[xn],Lv=$e(Qe.keys),Bv=$e(Qe.values),Fv=$e(Qe.entries),Uv=$e(Qe.set),lc=$e(Qe.reverse),zv=$e(Qe.fill),Vv=$e(Qe.copyWithin),uc=$e(Qe.sort),tr=$e(Qe.slice),Gv=$e(Qe.subarray),Ke=Ks(Qe,"buffer"),jn=Ks(Qe,"byteOffset"),_e=Ks(Qe,"length"),ep=Ks(Qe,Sl),Wv=Uint8Array,Ot=Uint16Array,cc=(...s)=>vt(Pv,Ot,s),Tl=Uint32Array,qv=Float32Array,ds=Br([][xn]()),To=$e(ds.next),Hv=$e((function*(){})().next),jv=Br(ds),De=TypeError,Qo=RangeError,tp=WeakSet,np=tp.prototype,Kv=$e(np.add),Xv=$e(np.has),Eo=WeakMap,El=Eo.prototype,Xi=$e(El.get),Yv=$e(El.has),Al=$e(El.set),sp=new Eo,Qv=So(null,{next:{value:function(){const e=Xi(sp,this);return To(e)}},[xn]:{value:function(){return this}}});function ei(s){if(s[xn]===Il&&ds.next===To)return s;const e=So(Qv);return Al(sp,e,Cv(s)),e}const rp=new Eo,ip=So(jv,{next:{value:function(){const e=Xi(rp,this);return Hv(e)},writable:!0,configurable:!0}});for(const s of jd(ds))s!=="next"&&Fr(ip,s,Ms(ds,s));function hc(s){const e=So(ip);return Al(rp,e,s),e}function Yi(s){return s!==null&&typeof s=="object"||typeof s=="function"}function fc(s){return s!==null&&typeof s=="object"}function Qi(s){return ep(s)!==void 0}function Oa(s){const e=ep(s);return e==="BigInt64Array"||e==="BigUint64Array"}function Zv(s){try{return Qd(s)?!1:(Ov(s),!0)}catch{return!1}}function op(s){if(Da===null)return!1;try{return Mv(s),!0}catch{return!1}}function Jv(s){return Zv(s)||op(s)}function dc(s){return Qd(s)?s[xn]===Il&&ds.next===To:!1}function e_(s){return Qi(s)?s[xn]===Rv&&ds.next===To:!1}function ti(s){if(typeof s!="string")return!1;const e=+s;return s!==e+""||!Xd(e)?!1:e===Zd(e)}const Zi=Sv("__Float16Array__");function t_(s){if(!fc(s))return!1;const e=Br(s);if(!fc(e))return!1;const t=e.constructor;if(t===void 0)return!1;if(!Yi(t))throw De(Hd);return $a(t,Zi)}const Ma=1/vv;function n_(s){return s+Ma-Ma}const ap=6103515625e-14,s_=65504,lp=.0009765625,pc=lp*ap,r_=lp*Ma;function i_(s){const e=+s;if(!Xd(e)||e===0)return e;const t=e>0?1:-1,n=$v(e);if(n<ap)return t*n_(n/pc)*pc;const r=(1+r_)*n,i=r-(r-n);return i>s_||Ps(i)?t*(1/0):t*i}const up=new ko(4),cp=new qv(up),hp=new Tl(up),Qt=new Ot(512),Zt=new Wv(512);for(let s=0;s<256;++s){const e=s-127;e<-24?(Qt[s]=0,Qt[s|256]=32768,Zt[s]=24,Zt[s|256]=24):e<-14?(Qt[s]=1024>>-e-14,Qt[s|256]=1024>>-e-14|32768,Zt[s]=-e-1,Zt[s|256]=-e-1):e<=15?(Qt[s]=e+15<<10,Qt[s|256]=e+15<<10|32768,Zt[s]=13,Zt[s|256]=13):e<128?(Qt[s]=31744,Qt[s|256]=64512,Zt[s]=24,Zt[s|256]=24):(Qt[s]=31744,Qt[s|256]=64512,Zt[s]=13,Zt[s|256]=13)}function un(s){cp[0]=i_(s);const e=hp[0],t=e>>23&511;return Qt[t]+((e&8388607)>>Zt[t])}const Cl=new Tl(2048);for(let s=1;s<1024;++s){let e=s<<13,t=0;for(;(e&8388608)===0;)e<<=1,t-=8388608;e&=-8388609,t+=947912704,Cl[s]=e|t}for(let s=1024;s<2048;++s)Cl[s]=939524096+(s-1024<<13);const Xs=new Tl(64);for(let s=1;s<31;++s)Xs[s]=s<<23;Xs[31]=1199570944;Xs[32]=2147483648;for(let s=33;s<63;++s)Xs[s]=2147483648+(s-32<<23);Xs[63]=3347054592;const fp=new Ot(64);for(let s=1;s<64;++s)s!==32&&(fp[s]=1024);function Ie(s){const e=s>>10;return hp[0]=Cl[fp[e]+(s&1023)]+Xs[e],cp[0]}function Sn(s){const e=+s;return Ps(e)||e===0?0:Zd(e)}function Zo(s){const e=Sn(s);return e<0?0:e<oc?e:oc}function ni(s,e){if(!Yi(s))throw De(mv);const t=s.constructor;if(t===void 0)return e;if(!Yi(t))throw De(Hd);const n=t[_v];return n??e}function dr(s){if(op(s))return!1;try{return Dv(s,0,0),!1}catch{}return!0}function mc(s,e){const t=Ps(s),n=Ps(e);if(t&&n)return 0;if(t)return 1;if(n||s<e)return-1;if(s>e)return 1;if(s===0&&e===0){const r=ac(s,0),i=ac(e,0);if(!r&&i)return-1;if(r&&!i)return 1}return 0}const $l=2,Ji=new Eo;function Ts(s){return Yv(Ji,s)||!Nv(s)&&t_(s)}function xe(s){if(!Ts(s))throw De(gv)}function si(s,e){const t=Ts(s),n=Qi(s);if(!t&&!n)throw De(yv);if(typeof e=="number"){let r;if(t){const i=de(s);r=_e(i)}else r=_e(s);if(r<e)throw De(bv)}if(Oa(s))throw De(Ca)}function de(s){const e=Xi(Ji,s);if(e!==void 0){const r=Ke(e);if(dr(r))throw De(fr);return e}const t=s.buffer;if(dr(t))throw De(fr);const n=ir(Ne,[t,s.byteOffset,s.length],s.constructor);return Xi(Ji,n)}function gc(s){const e=_e(s),t=[];for(let n=0;n<e;++n)t[n]=Ie(s[n]);return t}const dp=new tp;for(const s of jd(Qe)){if(s===Sl)continue;const e=Ms(Qe,s);An(e,"get")&&typeof e.get=="function"&&Kv(dp,e.get)}const o_=Iv({get(s,e,t){return ti(e)&&An(s,e)?Ie(Yo(s,e)):Xv(dp,kv(s,e))?Yo(s,e):Yo(s,e,t)},set(s,e,t,n){return ti(e)&&An(s,e)?ic(s,e,un(t)):ic(s,e,t,n)},getOwnPropertyDescriptor(s,e){if(ti(e)&&An(s,e)){const t=Ms(s,e);return t.value=Ie(t.value),t}return Ms(s,e)},defineProperty(s,e,t){return ti(e)&&An(s,e)&&An(t,"value")&&(t.value=un(t.value)),rc(s,e,t)}});class Ne{constructor(e,t,n){let r;if(Ts(e))r=ir(Ot,[de(e)],new.target);else if(Yi(e)&&!Jv(e)){let o,a;if(Qi(e)){o=e,a=_e(e);const l=Ke(e);if(dr(l))throw De(fr);if(Oa(e))throw De(Ca);const u=new ko(a*$l);r=ir(Ot,[u],new.target)}else{const l=e[xn];if(l!=null&&typeof l!="function")throw De(nc);l!=null?dc(e)?(o=e,a=e.length):(o=[...e],a=o.length):(o=e,a=Zo(o.length)),r=ir(Ot,[a],new.target)}for(let l=0;l<a;++l)r[l]=un(o[l])}else r=ir(Ot,arguments,new.target);const i=new xv(r,o_);return Al(Ji,i,r),i}static from(e,...t){const n=this;if(!$a(n,Zi))throw De(tc);if(n===Ne){if(Ts(e)&&t.length===0){const c=de(e),h=new Ot(Ke(c),jn(c),_e(c));return new Ne(Ke(tr(h)))}if(t.length===0)return new Ne(Ke(cc(e,un)));const l=t[0],u=t[1];return new Ne(Ke(cc(e,function(c,...h){return un(vt(l,this,[c,...ei(h)]))},u)))}let r,i;const o=e[xn];if(o!=null&&typeof o!="function")throw De(nc);if(o!=null)dc(e)?(r=e,i=e.length):e_(e)?(r=e,i=_e(e)):(r=[...e],i=r.length);else{if(e==null)throw De(Aa);r=Rs(e),i=Zo(r.length)}const a=new n(i);if(t.length===0)for(let l=0;l<i;++l)a[l]=r[l];else{const l=t[0],u=t[1];for(let c=0;c<i;++c)a[c]=vt(l,u,[r[c],c])}return a}static of(...e){const t=this;if(!$a(t,Zi))throw De(tc);const n=e.length;if(t===Ne){const i=new Ne(n),o=de(i);for(let a=0;a<n;++a)o[a]=un(e[a]);return i}const r=new t(n);for(let i=0;i<n;++i)r[i]=e[i];return r}keys(){xe(this);const e=de(this);return Lv(e)}values(){xe(this);const e=de(this);return hc((function*(){for(const t of Bv(e))yield Ie(t)})())}entries(){xe(this);const e=de(this);return hc((function*(){for(const[t,n]of Fv(e))yield[t,Ie(n)]})())}at(e){xe(this);const t=de(this),n=_e(t),r=Sn(e),i=r>=0?r:n+r;if(!(i<0||i>=n))return Ie(t[i])}with(e,t){xe(this);const n=de(this),r=_e(n),i=Sn(e),o=i>=0?i:r+i,a=+t;if(o<0||o>=r)throw Qo(Xo);const l=new Ot(Ke(n),jn(n),_e(n)),u=new Ne(Ke(tr(l))),c=de(u);return c[o]=un(a),u}map(e,...t){xe(this);const n=de(this),r=_e(n),i=t[0],o=ni(n,Ne);if(o===Ne){const l=new Ne(r),u=de(l);for(let c=0;c<r;++c){const h=Ie(n[c]);u[c]=un(vt(e,i,[h,c,this]))}return l}const a=new o(r);si(a,r);for(let l=0;l<r;++l){const u=Ie(n[l]);a[l]=vt(e,i,[u,l,this])}return a}filter(e,...t){xe(this);const n=de(this),r=_e(n),i=t[0],o=[];for(let u=0;u<r;++u){const c=Ie(n[u]);vt(e,i,[c,u,this])&&Ev(o,c)}const a=ni(n,Ne),l=new a(o);return si(l),l}reduce(e,...t){xe(this);const n=de(this),r=_e(n);if(r===0&&t.length===0)throw De(sc);let i,o;t.length===0?(i=Ie(n[0]),o=1):(i=t[0],o=0);for(let a=o;a<r;++a)i=e(i,Ie(n[a]),a,this);return i}reduceRight(e,...t){xe(this);const n=de(this),r=_e(n);if(r===0&&t.length===0)throw De(sc);let i,o;t.length===0?(i=Ie(n[r-1]),o=r-2):(i=t[0],o=r-1);for(let a=o;a>=0;--a)i=e(i,Ie(n[a]),a,this);return i}forEach(e,...t){xe(this);const n=de(this),r=_e(n),i=t[0];for(let o=0;o<r;++o)vt(e,i,[Ie(n[o]),o,this])}find(e,...t){xe(this);const n=de(this),r=_e(n),i=t[0];for(let o=0;o<r;++o){const a=Ie(n[o]);if(vt(e,i,[a,o,this]))return a}}findIndex(e,...t){xe(this);const n=de(this),r=_e(n),i=t[0];for(let o=0;o<r;++o){const a=Ie(n[o]);if(vt(e,i,[a,o,this]))return o}return-1}findLast(e,...t){xe(this);const n=de(this),r=_e(n),i=t[0];for(let o=r-1;o>=0;--o){const a=Ie(n[o]);if(vt(e,i,[a,o,this]))return a}}findLastIndex(e,...t){xe(this);const n=de(this),r=_e(n),i=t[0];for(let o=r-1;o>=0;--o){const a=Ie(n[o]);if(vt(e,i,[a,o,this]))return o}return-1}every(e,...t){xe(this);const n=de(this),r=_e(n),i=t[0];for(let o=0;o<r;++o)if(!vt(e,i,[Ie(n[o]),o,this]))return!1;return!0}some(e,...t){xe(this);const n=de(this),r=_e(n),i=t[0];for(let o=0;o<r;++o)if(vt(e,i,[Ie(n[o]),o,this]))return!0;return!1}set(e,...t){xe(this);const n=de(this),r=Sn(t[0]);if(r<0)throw Qo(Xo);if(e==null)throw De(Aa);if(Oa(e))throw De(Ca);if(Ts(e))return Uv(de(this),de(e),r);if(Qi(e)){const l=Ke(e);if(dr(l))throw De(fr)}const i=_e(n),o=Rs(e),a=Zo(o.length);if(r===1/0||a+r>i)throw Qo(Xo);for(let l=0;l<a;++l)n[l+r]=un(o[l])}reverse(){xe(this);const e=de(this);return lc(e),this}toReversed(){xe(this);const e=de(this),t=new Ot(Ke(e),jn(e),_e(e)),n=new Ne(Ke(tr(t))),r=de(n);return lc(r),n}fill(e,...t){xe(this);const n=de(this);return zv(n,un(e),...ei(t)),this}copyWithin(e,t,...n){xe(this);const r=de(this);return Vv(r,e,t,...ei(n)),this}sort(e){xe(this);const t=de(this),n=e!==void 0?e:mc;return uc(t,(r,i)=>n(Ie(r),Ie(i))),this}toSorted(e){xe(this);const t=de(this);if(e!==void 0&&typeof e!="function")throw new De(wv);const n=e!==void 0?e:mc,r=new Ot(Ke(t),jn(t),_e(t)),i=new Ne(Ke(tr(r))),o=de(i);return uc(o,(a,l)=>n(Ie(a),Ie(l))),i}slice(e,t){xe(this);const n=de(this),r=ni(n,Ne);if(r===Ne){const I=new Ot(Ke(n),jn(n),_e(n));return new Ne(Ke(tr(I,e,t)))}const i=_e(n),o=Sn(e),a=t===void 0?i:Sn(t);let l;o===-1/0?l=0:o<0?l=i+o>0?i+o:0:l=i<o?i:o;let u;a===-1/0?u=0:a<0?u=i+a>0?i+a:0:u=i<a?i:a;const c=u-l>0?u-l:0,h=new r(c);if(si(h,c),c===0)return h;const d=Ke(n);if(dr(d))throw De(fr);let w=0;for(;l<u;)h[w]=Ie(n[l]),++l,++w;return h}subarray(e,t){xe(this);const n=de(this),r=ni(n,Ne),i=new Ot(Ke(n),jn(n),_e(n)),o=Gv(i,e,t),a=new r(Ke(o),jn(o),_e(o));return si(a),a}indexOf(e,...t){xe(this);const n=de(this),r=_e(n);let i=Sn(t[0]);if(i===1/0)return-1;i<0&&(i+=r,i<0&&(i=0));for(let o=i;o<r;++o)if(An(n,o)&&Ie(n[o])===e)return o;return-1}lastIndexOf(e,...t){xe(this);const n=de(this),r=_e(n);let i=t.length>=1?Sn(t[0]):r-1;if(i===-1/0)return-1;i>=0?i=i<r-1?i:r-1:i+=r;for(let o=i;o>=0;--o)if(An(n,o)&&Ie(n[o])===e)return o;return-1}includes(e,...t){xe(this);const n=de(this),r=_e(n);let i=Sn(t[0]);if(i===1/0)return!1;i<0&&(i+=r,i<0&&(i=0));const o=Ps(e);for(let a=i;a<r;++a){const l=Ie(n[a]);if(o&&Ps(l)||l===e)return!0}return!1}join(e){xe(this);const t=de(this),n=gc(t);return Tv(n,e)}toLocaleString(...e){xe(this);const t=de(this),n=gc(t);return Av(n,...ei(e))}get[Sl](){if(Ts(this))return"Float16Array"}}Fr(Ne,"BYTES_PER_ELEMENT",{value:$l});Fr(Ne,Zi,{});Kd(Ne,kl);const eo=Ne.prototype;Fr(eo,"BYTES_PER_ELEMENT",{value:$l});Fr(eo,xn,{value:eo.values,writable:!0,configurable:!0});Kd(eo,Qe);function a_(s,e){return s.channels===e.channels}const ri=8;class ii{autoUpdateOutputBuffer=!0;_label;_device;_outputBuffers={};_pipeline;_bindGroups=[];_needsUpdatePipeline=!0;_needsResizeBuffer=!0;_inputs=[];_outputs=[];_uniforms=[];_uniformBuffers={};_width=10;_height=10;_execWidth;_execHeight;_csCode="";_csMain;_csDefine;_groupOffsets={inputs:0,uniforms:1,outputs:2};constructor(e,t,n){this._label=e,this._device=t,this._csMain=n.csMain,this._csDefine=n.csDefine,this._inputs=n.inputs,this._outputs=n.outputs,this._uniforms=n.uniforms,this.autoUpdateOutputBuffer=n.autoUpdateOutputBuffer??!0,n.uniforms.forEach(r=>{this._uniformBuffers[r.label]=t.createBuffer({label:this._label,size:r.data.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this._device.queue.writeBuffer(this._uniformBuffers[r.label],0,r.data)})}setCSCode({csDefine:e,csMain:t}){this._csDefine=e,this._csMain=t,this._needsUpdatePipeline=!0}setSize(e,t){e=Math.ceil(e),t=Math.ceil(t);const n=e!==this._width||t!==this._height;this._width=e,this._height=t,n&&(this._needsResizeBuffer=!0,this._needsUpdatePipeline=!0)}setExecuteSize(e,t){e=Math.ceil(e),t=Math.ceil(t),this._execWidth=e,this._execHeight=t}setOutputParams(e){this.autoUpdateOutputBuffer&&this._updateOutputBuffers(e),this._needsUpdatePipeline=!0}setOutputBuffers(e){this._outputBuffers=Object.keys(e).reduce((t,n)=>(t[n]={buffer:e[n],params:{channels:4}},t),{})}setUniform(e,t){const n=this._uniformBuffers[e];this._device.queue.writeBuffer(n,0,t)}getOutput(e){return this._needsResizeBuffer&&this.autoUpdateOutputBuffer&&(this._resizeOutputBuffers(),this._needsResizeBuffer=!1),this._outputBuffers[e].buffer}dispose(){Object.keys(this._uniformBuffers).forEach(e=>{this._uniformBuffers[e].destroy()}),Object.keys(this._outputBuffers).forEach(e=>{this._outputBuffers[e].buffer.destroy()})}_createBuffer(e){const t=this._width*this._height*4*4;return this._device.createBuffer({label:this._label,size:Math.max(t,80),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC})}_resizeOutputBuffers(){const e=this._outputBuffers;for(const t in e){const{buffer:n,params:r}=e[t];n.destroy(),e[t].buffer=this._createBuffer(r)}}_updateOutputBuffers(e){const t=this._outputBuffers;for(const n in e){const r=e[n];if(!a_(r,t[n]?.params||{})){t[n]?.buffer.destroy();const i=this._createBuffer(r);t[n]={buffer:i,params:r}}}}_updatePipeline(e,t){if(!this._needsUpdatePipeline)return;this._needsUpdatePipeline=!1;const n=this._device,r=this._getFullCs(e,t);r!==this._csCode&&(this._csCode=r,this._pipeline=n.createComputePipeline({label:this._label,layout:"auto",compute:{module:n.createShaderModule({label:this._label,code:r}),entryPoint:"main"}}),this._updateBindGroups())}_getFullCs(e,t){const n=this._inputs,r=this._uniforms;let i=0;const o=this._groupOffsets={inputs:0,uniforms:0,outputs:0};return n.length>0&&i++,r.length>0&&(o.uniforms=i,i++),o.outputs=i,`
${n.sort().map((l,u)=>{const c=`@group(${o.inputs}) @binding(${u}) `,h=`in_${l}`;return t[l]==="texture"?`${c} var ${h}: texture_2d<f32>;`:`${c} var<storage, read> ${h}: array<vec${e[l].channels}f>;`}).join(`
`)}
${this._uniforms.map((l,u)=>`@group(${o.uniforms}) @binding(${u}) var<uniform> ${l.label}: ${l.type};`).join(`
`)}

${this._outputs.map((l,u)=>`@group(${o.outputs}) @binding(${u}) var<storage, read_write> out_${l}: array<vec${this._outputBuffers[l].params.channels}f>;`).join(`
`)}
${this._csDefine??""}
@compute @workgroup_size(${ri}, ${ri}, 1)
fn main(@builtin(global_invocation_id) globalId: vec3u) {
${this._csMain}
}
`}_updateBindGroups(){const e=[],t=this._device,n=this._groupOffsets;this._uniforms.length>0&&(e[n.uniforms]=t.createBindGroup({label:this._label,layout:this._pipeline.getBindGroupLayout(n.uniforms),entries:this._uniforms.map((r,i)=>({binding:i,resource:{buffer:this._uniformBuffers[r.label]}}))})),this._bindGroups=e}createPass(e,t){this._needsResizeBuffer&&this.autoUpdateOutputBuffer&&(this._resizeOutputBuffers(),this._needsResizeBuffer=!1);const n=this._inputs.reduce((o,a)=>(o[a]=t[a].buffer?"buffer":"texture",o),{});this._updatePipeline(t,n);const r=this._groupOffsets;this._inputs.length>0&&(this._bindGroups[r.inputs]=this._device.createBindGroup({label:this._label,layout:this._pipeline.getBindGroupLayout(r.inputs),entries:this._inputs.map((o,a)=>({binding:a,resource:t[o].buffer?{buffer:t[o].buffer}:t[o].texture.createView()}))})),this._bindGroups[r.outputs]=this._device.createBindGroup({label:this._label,layout:this._pipeline.getBindGroupLayout(r.outputs),entries:this._outputs.map((o,a)=>({binding:a,resource:{buffer:this._outputBuffers[o].buffer}}))});const i=e.beginComputePass();i.setPipeline(this._pipeline),this._bindGroups.forEach((o,a)=>{i.setBindGroup(a,o)}),i.dispatchWorkgroups(Math.ceil((this._execWidth??this._width)/ri),Math.ceil((this._execHeight??this._height)/ri),1),i.end()}}const Nl=1412.83765,Dl=1.64593172,Ol=.431384981,Ml=-.00294139609,Pl=.192653254,Rl=.00626026094,Ll=.998620152,pp=15794576e-13,mp=.0322087631,gp=.00223151711,yp=.370974749;function bp(s){return s<=pp?s=Nl*s:s<=mp?s=Dl*Math.pow(s,Ol)+Ml:s=Pl*Math.log(s+Rl)+Ll,s}function l_(s){return s<=gp?s=s/Nl:s<=yp?s=Math.pow((s-Ml)/Dl,1/Ol):s=Math.exp((s-Ll)/Pl)-Rl,s}const u_=65504,wp=bp(u_),xp=1/wp,vp=wp;class Jo{x;y;width;height;constructor(e,t,n,r){this.x=e,this.y=t,this.width=n,this.height=r}}function c_({data:s,channels:e}){let t=0;for(let o=0;o<s.length;o+=e){const a=s[o],l=s[o+1],u=s[o+2],c=.212671*a+.71516*l+.072169*u;t+=Math.log2(c+1e-4)}const n=s.length/e,r=t/n;return .18/Math.pow(2,r)}function h_({data:s,channels:e,inputScale:t}){const n=new Float32Array(s.length);n.set(s);for(let r=0;r<n.length;r+=e)for(let i=0;i<3;i++){let o=n[r+i]*t;n[r+i]=bp(o)*xp}return n}function f_({data:s,channels:e,inputScale:t}){const n=new Float32Array(s.length);n.set(s);const r=1/t;for(let i=0;i<n.length;i+=e)for(let o=0;o<3;o++){let a=n[i+o]*vp;n[i+o]=l_(a)*r}return n}const yc=`
const a = ${Nl};
const b = ${Dl};
const c = ${Ol};
const d = ${Ml};
const e = ${Pl};
const f = ${Rl};
const g = ${Ll};
const y0 =${pp};
const y1 =${mp};
const x0 =${gp};
const x1 =${yp};

const normScale = ${xp};
const rcpNormScale = ${vp};
`;class d_{_device;_isHDR;_inputPassAux;_inputPassColor;_outputPass;_copyPass;_isInputTexture;constructor(e,t){this._device=e,this._isHDR=t;const n=[{label:"inputScale",type:"f32",data:new Float32Array([1])},{label:"inputSize",type:"vec2i",data:new Int32Array(2)},{label:"outputSize",type:"vec2i",data:new Int32Array(2)},{label:"inputOffset",type:"vec2i",data:new Int32Array(2)}];this._inputPassAux=new ii("inputPassAux",this._device,{inputs:["color","albedo","normal"],outputs:["color","albedo","normal"],uniforms:n,csDefine:"",csMain:""}),this._inputPassColor=new ii("inputPassColor",this._device,{inputs:["color"],outputs:["color"],uniforms:n,csDefine:"",csMain:""}),this._outputPass=new ii("outputPass",this._device,{inputs:["color","raw"],outputs:["color"],uniforms:[{label:"inputScale",type:"f32",data:new Float32Array([1])},{label:"inputSize",type:"vec2i",data:new Int32Array(2)},{label:"outputSize",type:"vec2i",data:new Int32Array(2)},{label:"imageSize",type:"vec2i",data:new Int32Array(2)},{label:"inputOffset",type:"vec2i",data:new Int32Array(2)},{label:"outputOffset",type:"vec2i",data:new Int32Array(2)}],csDefine:"",csMain:""}),this._copyPass=new ii("copyPass",this._device,{inputs:["color"],outputs:["color"],autoUpdateOutputBuffer:!1,uniforms:[{label:"size",type:"vec2i",data:new Int32Array(2)}],csMain:`
let outIdx = i32(globalId.x + globalId.y * u32(size.x));
out_color[outIdx] = textureLoad(in_color, globalId.xy, 0);
`}),this._inputPassAux.setOutputParams({color:{channels:3},albedo:{channels:3},normal:{channels:3}}),this._inputPassColor.setOutputParams({color:{channels:3}}),this._outputPass.setOutputParams({color:{channels:4}})}_updatePasses(e,t=!1){if(this._isInputTexture!=null&&this._isInputTexture===e)return;this._isInputTexture=e;const n=this._isHDR,r=`
${yc}
fn PUForward(y: f32) -> f32 {
  if (y <= y0) {
    return a * y;
  } else if (y <= y1) {
    return b * pow(y, c) + d;
  } else {
    return e * log(y + f) + g;
  }
}`;function i(a){return e?`textureLoad(in_${a}, globalId.xy + vec2u(inputOffset), 0)`:`in_${a}[inIdx]`}const o=`
let x = i32(globalId.x);
let y = i32(globalId.y);
let inIdx = (y + inputOffset.y) * inputSize.x + (x + inputOffset.x);
let col = ${i("color")};

let outIdx = y * outputSize.x + x;

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
${yc}
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
let x = i32(globalId.x);
let y = i32(globalId.y);
if (x >= outputSize.x || y >= outputSize.y) {
  return;
}
let inIdx = (y + inputOffset.y) * inputSize.x + x + inputOffset.x;
let outIdx = (y + outputOffset.y) * imageSize.x + x + outputOffset.x;
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
`})}setImageSize(e,t){this._inputPassAux.setUniform("inputSize",new Int32Array([e,t])),this._inputPassColor.setUniform("inputSize",new Int32Array([e,t])),this._outputPass.setUniform("imageSize",new Int32Array([e,t])),this._outputPass.setSize(e,t),this._copyPass.setSize(e,t),this._copyPass.setUniform("size",new Int32Array([e,t]))}setInputTile(e){const t=new Int32Array([e.width,e.height]);[this._inputPassAux,this._inputPassColor].forEach(n=>{n.setUniform("inputOffset",new Int32Array([e.x,e.y])),n.setUniform("outputSize",t),n.setSize(t[0],t[1])}),this._outputPass.setUniform("inputSize",t)}setOutputTile(e,t){const n=this._outputPass,r=new Int32Array([e.width,e.height]),i=e.x-t.x,o=e.y-t.y;n.setUniform("outputSize",r),n.setUniform("inputOffset",new Int32Array([i,o])),n.setUniform("outputOffset",new Int32Array([e.x,e.y])),n.setExecuteSize(r[0],r[1])}forward(e,t,n,r){const i=e instanceof GPUTexture;this._updatePasses(i,r);const o=this._inputPassAux,a=this._inputPassColor,l=this._device.createCommandEncoder();function u(c){return c instanceof GPUTexture?{texture:c,channels:4}:{buffer:c,channels:4}}return t&&n?o.createPass(l,{color:u(e),albedo:u(t),normal:u(n)}):a.createPass(l,{color:u(e)}),this._device.queue.submit([l.finish()]),t&&n?{color:o.getOutput("color"),albedo:o.getOutput("albedo"),normal:o.getOutput("normal")}:{color:a.getOutput("color")}}inverse(e,t){const r=this._device.createCommandEncoder(),i=this._outputPass;return i.createPass(r,{color:{buffer:e,channels:4},raw:t instanceof GPUBuffer?{buffer:t,channels:4}:{texture:t,channels:4}}),this._device.queue.submit([r.finish()]),i.getOutput("color")}copyInputDataToOutput(e){const t=this._device.createCommandEncoder(),r=this._outputPass.getOutput("color"),i=this._copyPass;e instanceof GPUTexture?(i.setOutputBuffers({color:r}),i.createPass(t,{color:{texture:e,channels:4}})):t.copyBufferToBuffer(e,0,r,0,r.size),this._device.queue.submit([t.finish()])}dispose(){this._outputPass.dispose(),this._inputPassAux.dispose()}}function bc(s,e){const t=s.buffer;if(e==="Float32")return new Float32Array(s.buffer);const n=new Ne(t),r=new Float32Array(n.length);for(let i=0;i<r.length;++i)r[i]=n[i];return r}function p_(s,e){const[t,n,r,i]=e,o=new Float32Array(s.length);for(let a=0;a<t;++a)for(let l=0;l<n;++l)for(let u=0;u<r;++u)for(let c=0;c<i;++c){const h=a*n*r*i+l*r*i+u*i+c,d=u*i*n*t+c*n*t+l*t+a;o[d]=s[h]}return o}function pr(s,e){return Math.ceil(s/e)*e}function oi(s){return s.data instanceof GPUBuffer||s.data instanceof GPUTexture}const m_=174,g_=202,_p=16,ai=pr(m_/2,_p),wc=pr(g_/2,_p);class y_{_hostTensors;_backend;_tfModel;_device;_tileWidth=0;_tileHeight=0;_tileOverlapX=0;_tileOverlapY=0;_aux;_hdr;_dataProcessGPU;_maxTileSize;_tensors=new Map;_modelsCache=new Map;constructor(e,t,n={}){this._hostTensors=e,this._backend=t,this._aux=n.aux||!1,this._hdr=n.hdr||!1,this._maxTileSize=pr(n.maxTileSize??512,2),this._device=this._backend.device}getDevice(){return this._device}_buildModel(e){const n=3+(this._aux?6:0),r=this._getTileSizeWithOverlap(),i=this._modelsCache,o=[r.width,r.height].join(",");if(i.has(o)){this._tfModel=i.get(o);return}const a=Y2({name:"input",shape:[r.height,r.width,n],dtype:"float32"});this._tfModel=new _o({inputs:[a],outputs:e?this._addNetLarge(a):this._addNet(a)}),i.set(o,this._tfModel)}_createConv(e,t,n){const r=e+".weight",i=e+".bias",o=this._tensors;let a=o.get(r),l=o.get(i);const u=this._hostTensors.get(r);if(!a){const h=u.desc.dims;a=fi(p_(bc(u.data,u.desc.dataType),h),[h[2],h[3],h[1],h[0]],"float32"),o.set(r,a)}if(!l){const h=this._hostTensors.get(e+".bias");l=pt(bc(h.data,h.desc.dataType),"float32"),o.set(i,l)}return new js({name:e,filters:u.desc.dims[0],kernelSize:u.desc.dims.slice(2,4),useBias:!0,activation:n,padding:"same",weights:[a,l],trainable:!1}).apply(t)}_createConcatConv(e,t,n){const r=new _l({name:e+"/concat",trainable:!1,axis:3});return this._createConv(e,r.apply([t,n]),"relu")}_createPooling(e){return new wl({name:e.name+"/pooling",poolSize:[2,2],strides:[2,2],padding:"same",trainable:!1}).apply(e)}_addUpsamplingLayer(e){return new bl({name:e.name+"/upsampling",size:[2,2],trainable:!1}).apply(e)}_addNet(e){let t=this._createConv("enc_conv0",e,"relu");const n=t=this._createPooling(this._createConv("enc_conv1",t,"relu")),r=t=this._createPooling(this._createConv("enc_conv2",t,"relu")),i=t=this._createPooling(this._createConv("enc_conv3",t,"relu")),o=t=this._createPooling(this._createConv("enc_conv4",t,"relu"));return t=this._createConv("enc_conv5a",o,"relu"),t=this._addUpsamplingLayer(this._createConv("enc_conv5b",t,"relu")),t=this._createConcatConv("dec_conv4a",t,i),t=this._addUpsamplingLayer(this._createConv("dec_conv4b",t,"relu")),t=this._createConcatConv("dec_conv3a",t,r),t=this._addUpsamplingLayer(this._createConv("dec_conv3b",t,"relu")),t=this._createConcatConv("dec_conv2a",t,n),t=this._addUpsamplingLayer(this._createConv("dec_conv2b",t,"relu")),t=this._createConcatConv("dec_conv1a",t,e),t=this._createConv("dec_conv1b",t,"relu"),t=this._createConv("dec_conv0",t,"relu"),t}_addNetLarge(e){let t=this._createConv("enc_conv1a",e,"relu");const n=t=this._createPooling(this._createConv("enc_conv1b",t,"relu"));t=this._createConv("enc_conv2a",t,"relu");const r=t=this._createPooling(this._createConv("enc_conv2b",t,"relu"));t=this._createConv("enc_conv3a",t,"relu");const i=t=this._createPooling(this._createConv("enc_conv3b",t,"relu"));t=this._createConv("enc_conv4a",t,"relu");const o=t=this._createPooling(this._createConv("enc_conv4b",t,"relu"));return t=this._createConv("enc_conv5a",o,"relu"),t=this._addUpsamplingLayer(this._createConv("enc_conv5b",t,"relu")),t=this._createConcatConv("dec_conv4a",t,i),t=this._addUpsamplingLayer(this._createConv("dec_conv4b",t,"relu")),t=this._createConcatConv("dec_conv3a",t,r),t=this._addUpsamplingLayer(this._createConv("dec_conv3b",t,"relu")),t=this._createConcatConv("dec_conv2a",t,n),t=this._addUpsamplingLayer(this._createConv("dec_conv2b",t,"relu")),t=this._createConcatConv("dec_conv1a",t,e),t=this._createConv("dec_conv1b",t,"relu"),t=this._createConv("dec_conv1c",t,"relu"),t}_updateModel(e,t){const n=this._hostTensors.has("enc_conv1b.weight"),r=this._maxTileSize;let i=r,o=r,a=n?wc:ai,l=n?wc:ai;e<r+ai*2&&(i=pr(e,r/2),e<=r&&(a=0)),t<r+ai*2&&(o=pr(t,r/2),t<=r&&(l=0));const u=Math.max(i,o),c=Math.max(a,l);i=u,o=u,a=c,l=c,(i!==this._tileWidth||o!==this._tileHeight||a!==this._tileOverlapX||l!==this._tileOverlapY||!this._tfModel)&&(this._tileWidth=i,this._tileHeight=o,this._tileOverlapX=a,this._tileOverlapY=l,this._buildModel(n))}_getTileSizeWithOverlap(){return{width:this._tileWidth+2*this._tileOverlapX,height:this._tileHeight+2*this._tileOverlapY}}_processImageData(e,t,n,r){const i=e.data,o=i.length/4,a=this._aux?9:3,l=new Float32Array(o*a);if(t&&!n||n&&!t)throw new Error("Normal map and albedo map are both required");if(t&&n&&(t.width!==n.width||t.height!==n.height||e.width!==t.width||e.height!==t.height))throw new Error("Image size mismatch");const u=t?.data,c=n?.data;for(let h=0;h<i.length;h+=4){const d=h/4*a;for(let w=0;w<3;w++)r?l[d+w]=i[h+w]:l[d+w]=i[h+w]/255,u&&(l[d+w+3]=u[h+w]/255),c&&(l[d+w+6]=c[h+w]/255)}return l}_readTile(e,t,n,r){const i=new Float32Array(n.width*n.height*t);for(let o=0;o<n.height;o++)for(let a=0;a<n.width;a++){const l=((o+n.y)*r+(a+n.x))*t,u=(o*n.width+a)*t;for(let c=0;c<t;c++)i[u+c]=e[l+c]}return i}_writeTile(e,t,n,r,i,o){const{data:a,width:l}=e,u=n.x-t.x,c=n.y-t.y;for(let h=0;h<n.height;h++)for(let d=0;d<n.width;d++){const w=((h+c)*i+d+u)*3,I=((h+n.y)*l+(d+n.x))*4;for(let E=0;E<3;E++)o?a[I+E]=r[w+E]:a[I+E]=Math.min(Math.max(r[w+E]*255,0),255);e.data[I+3]=o?1:255}}_executeTile(e,t,n,r,i,o,a,l,u){const c=this._aux?9:3,h=this._tileOverlapX,d=this._tileOverlapY;let w=this._getTileSizeWithOverlap(),I={width:this._tileWidth,height:this._tileHeight},E=r>0?r*I.width-h:0,m=Math.min(E+w.width,o);E=Math.max(m-w.width,0);let S=i>0?i*I.height-d:0,b=Math.min(S+w.height,a);S=Math.max(b-w.height,0);const f=w.width,_=w.height,v=new Jo(E,S,f,_);let T,N=1;const O=this._device;let $=this._dataProcessGPU;if(e instanceof Float32Array){let k=this._readTile(e,c,v,o);l&&(N=c_({data:k,channels:c}),k=h_({data:k,channels:c,inputScale:N})),T=fi(k,[1,_,f,c],"float32")}else{$||($=this._dataProcessGPU=new d_(O,l)),$.setImageSize(o,a),$.setInputTile(v),r===0&&i===0&&$.copyInputDataToOutput(e.color);const{color:k,albedo:C,normal:R}=$.forward(e.color,this._aux?e.albedo:void 0,this._aux?e.normal:void 0,u),z=j=>{const G=fi({buffer:j,zeroCopy:!0},[1,_,f,4]);return yr(G,[0,0,0,0],[1,_,f,3])};if(this._aux){const j=[k,C,R].map(G=>z(G));T=Uy(j,3)}else T=z(k)}let A;const g=this._tfModel.predict(T),p=Math.min(I.width,o),y=Math.min(I.height,a),x=new Jo(r*p,i*y,p,y);if(x.width=Math.min(x.width,o-x.x),x.height=Math.min(x.height,a-x.y),e instanceof Float32Array){let k=g.dataSync();l&&(k=f_({data:k,channels:3,inputScale:N})),this._writeTile(n,v,x,k,w.width,l);for(let C=0;C<y;C++)for(let R=0;R<p;R++){const z=(C*p+R)*4,j=((C+x.y)*o+(R+x.x))*4;for(let G=0;G<4;G++)t.data[z+G]=n.data[j+G]}}else{$.setOutputTile(x,v);const k=My(g,[[0,0],[0,0],[0,0],[0,1]]);A=$.inverse(k.dataToGPU().buffer,e.color)}return A}tileExecute({color:e,albedo:t,normal:n,done:r,progress:i,denoiseAlpha:o}){if(this._aux&&(!t||!n))throw new Error("Normal map and albedo map are both required");if(!this._aux&&(t||n))throw new Error("Normal map and albedo map are not required");const a=e.width,l=e.height;this._updateModel(a,l);const u=this._hdr||!1;let c;oi(e)||(c=this._processImageData(e,t,n,u));const h=this._tileWidth,d=this._tileHeight,w=Math.ceil(l/d),I=Math.ceil(a/h);function E(_,v){return u?{data:new Float32Array(_*v*4),width:_,height:v}:new ImageData(_,v)}const m=oi(e)?void 0:E(a,l),S=oi(e)?void 0:E(Math.min(h,a),Math.min(d,l));let b=!1;const f=(_,v)=>{if(b)return;let T;H.startScope(),T=this._executeTile(oi(e)?{color:e.data,albedo:t?.data,normal:n?.data}:c,S,m,_,v,a,l,u,o),H.endScope();const N=m||{data:T,width:a,height:l};i?.(N,S,new Jo(_*h,v*d,h,d),_+v*I,I*w),_+1<I||v+1<w?requestAnimationFrame(()=>{_+1<I?f(_+1,v):v+1<w&&f(0,v+1)}):r(N)};return f(0,0),()=>{b=!0}}dispose(){this._tfModel?.dispose(),this._dataProcessGPU?.dispose(),this._tensors.forEach(e=>e.dispose())}}/**
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
 */const Lt=fe();Lt.registerFlag("WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE",()=>15);Lt.registerFlag("WEBGPU_CPU_FORWARD",()=>!0);Lt.registerFlag("WEBGPU_MATMUL_PROGRAM_TYPE",()=>-1);Lt.registerFlag("WEBGPU_USE_NAIVE_CONV2D_TRANSPOSE",()=>!0);Lt.registerFlag("WEBGPU_USE_LOW_POWER_GPU",()=>!1);Lt.registerFlag("WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD",()=>1e3);Lt.registerFlag("WEBGPU_USE_PROFILE_TOOL",()=>!1);Lt.registerFlag("WEBGPU_IMPORT_EXTERNAL_TEXTURE",()=>!0);Lt.registerFlag("WEBGPU_USE_NAIVE_CONV2D_DEBUG",()=>!1);Lt.registerFlag("WEBGPU_THRESHOLD_TO_INCREASE_WORKGROUPS_FOR_MATMUL",()=>-1);Lt.registerFlag("WEBGPU_CONV_SEPARATE_IM2COL_SHADER",()=>!1);Lt.registerFlag("WEBGPU_PRINT_SHADER",()=>"");Lt.registerFlag("WEBGPU_ENGINE_COMPILE_ONLY",()=>!1);/**
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
 */class b_{constructor(e){e&&(this.vendor=e.vendor,this.architecture=e.architecture,this.intelGPUGeneration=this.getIntelGPUGeneration())}getIntelGPUGeneration(){if(this.isIntel()){if(this.architecture.startsWith("gen"))return Number(this.architecture.match(/\d+/));if(this.architecture.startsWith("xe"))return 12}return 0}isIntel(){return this.vendor==="intel"}}/**
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
 */class w_{constructor(e){this.device=e,this.numUsedBuffers=0,this.numFreeBuffers=0,this.freeBuffers=new Map,this.usedBuffers=new Map,this.numBytesUsed=0,this.numBytesAllocated=0}acquireBuffer(e,t,n=!1,r=!0){let i;const o=xc(e,t);return r?(this.freeBuffers.has(o)||this.freeBuffers.set(o,[]),this.freeBuffers.get(o).length>0?(i=this.freeBuffers.get(o).pop(),this.numFreeBuffers--):(i=this.device.createBuffer({size:e,usage:t,mappedAtCreation:n}),this.numBytesAllocated+=e)):(i=this.device.createBuffer({size:e,usage:t,mappedAtCreation:n}),this.numBytesAllocated+=e),this.usedBuffers.has(o)||this.usedBuffers.set(o,[]),this.usedBuffers.get(o).push(i),this.numUsedBuffers++,this.numBytesUsed+=e,i}releaseBuffer(e,t=!0){if(this.freeBuffers.size===0)return;const n=e.size,r=e.usage,i=xc(n,r),o=this.usedBuffers.get(i),a=o.indexOf(e);if(a<0)throw new Error("Cannot find the buffer in buffer manager");o[a]=o[o.length-1],o.pop(),this.numUsedBuffers--,this.numBytesUsed-=n,t?(this.freeBuffers.get(i).push(e),this.numFreeBuffers++):(e.destroy(),this.numBytesAllocated-=n)}getNumUsedBuffers(){return this.numUsedBuffers}getNumFreeBuffers(){return this.numFreeBuffers}dispose(){this.freeBuffers.forEach((e,t)=>{e.forEach(n=>{n.destroy()})}),this.usedBuffers.forEach((e,t)=>{e.forEach(n=>{n.destroy()})}),this.freeBuffers=new Map,this.usedBuffers=new Map,this.numUsedBuffers=0,this.numFreeBuffers=0,this.numBytesUsed=0,this.numBytesAllocated=0}}function xc(s,e){return`${s}_${e}`}/**
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
 */class x_{constructor(e){this.device=e,this.numUsedTextures=0,this.numFreeTextures=0,this.freeTextures=new Map,this.usedTextures=new Map,this.numBytesUsed=0,this.numBytesAllocated=0}acquireTexture(e,t,n,r){const i=_c(n),o=e*t*i,a=vc(e,t,n,r);if(this.freeTextures.has(a)||this.freeTextures.set(a,[]),this.usedTextures.has(a)||this.usedTextures.set(a,[]),this.numBytesUsed+=o,this.numUsedTextures++,this.freeTextures.get(a).length>0){this.numFreeTextures--;const u=this.freeTextures.get(a).shift();return this.usedTextures.get(a).push(u),u}this.numBytesAllocated+=o;const l=this.device.createTexture({size:[e,t],format:n,usage:r});return this.usedTextures.get(a).push(l),l}releaseTexture(e){if(this.freeTextures.size===0)return;const t=e.width,n=e.height,r=e.format,i=e.usage,o=vc(t,n,r,i);this.freeTextures.has(o)||this.freeTextures.set(o,[]),this.freeTextures.get(o).push(e),this.numFreeTextures++,this.numUsedTextures--;const a=this.usedTextures.get(o),l=a.indexOf(e);if(l<0)throw new Error("Cannot release a texture that was never provided by this texture manager");a.splice(l,1);const u=_c(r),c=t*n*u;this.numBytesUsed-=c}getNumUsedTextures(){return this.numUsedTextures}getNumFreeTextures(){return this.numFreeTextures}dispose(){this.freeTextures.forEach((e,t)=>{e.forEach(n=>{n.destroy()})}),this.usedTextures.forEach((e,t)=>{e.forEach(n=>{n.destroy()})}),this.freeTextures=new Map,this.usedTextures=new Map,this.numUsedTextures=0,this.numFreeTextures=0,this.numBytesUsed=0,this.numBytesAllocated=0}}function vc(s,e,t,n){return`${s}_${e}_${t}_${n}`}function _c(s){if(s==="rgba8unorm")return 16;throw new Error(`${s} is not supported!`)}/**
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
 */function v_(s,e){if(Math.max(...s)>5)throw new Error("Cannot symbolically compute strides for rank > 6 tensor.");const t=s.length,n="xyzwuv",r=s.map(o=>`${e}.${n[o]}`),i=new Array(t-1);i[t-2]=r[t-1];for(let o=t-3;o>=0;--o)i[o]=`(${i[o+1]} * ${r[o+1]})`;return i}const __=(s,e,t)=>`
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
 */var to;(function(s){s[s.FROM_PIXELS=0]="FROM_PIXELS",s[s.DRAW=1]="DRAW"})(to||(to={}));const S_=(s,e,t,n,r)=>{const i={dtype:n.dtype,shape:n.shape},o=k_(t,i,e),a=s.createShaderModule({code:o,label:e.constructor.name});let l=fe().get("WEBGPU_PRINT_SHADER");if(l!==""){l=l.toLowerCase();const u=l.split(",");(l==="all"||u.some(c=>e.shaderKey.toLowerCase().includes(c)))&&(console.group(e.shaderKey),console.debug(o),console.groupEnd())}return r?s.createComputePipelineAsync({compute:{module:a,entryPoint:"_start"},label:e.constructor.name,layout:"auto"}):s.createComputePipeline({compute:{module:a,entryPoint:"_start"},label:e.constructor.name,layout:"auto"})},me=(s,e="f32")=>{switch(s){case 1:return`${e}`;case 2:return`vec2<${e}>`;case 3:return`vec3<${e}>`;case 4:return`vec4<${e}>`;default:throw new Error(`${s}-component ${e} is not supported.`)}};function ht(s){if(s<=1)return"i32";if(s===2)return"vec2<i32>";if(s===3)return"vec3<i32>";if(s===4)return"vec4<i32>";if(s===5)return"vec5";if(s===6)return"vec6";throw Error(`GPU for rank ${s} is not yet supported`)}function is(s){if(s===0)return"x";if(s===1)return"y";if(s===2)return"z";if(s===3)return"w";if(s===4)return"u";if(s===5)return"v";throw Error(`Index ${s} is not yet supported`)}function je(...s){let e;switch(s.length){case 0:e=`
        fn main()
      `;break;case 1:e=`
        fn main(${s[0]} : i32)
      `;break;default:throw Error("Unreachable")}return e}function Sc(s,e){let t;return t=`
     ${I_(e)}
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
    `,t}function I_(s){return`
  @compute @workgroup_size(${s.workgroupSize[0]}, ${s.workgroupSize[1]}, ${s.workgroupSize[2]})
`}function k_(s,e,t){const n=[],r=t.workgroupSize[0]*t.workgroupSize[1]*t.workgroupSize[2];if(t.outputComponent=t.outputComponent?t.outputComponent:1,n.push(`

      var<private> localId: vec3<u32>;
      var<private> localIndex: u32;
      var<private> globalId: vec3<u32>;
      var<private> numWorkgroups: vec3<u32>;
      var<private> workgroupId: vec3<u32>;

      // Only used when the y/z dimension of workgroup size is 1.
      fn getGlobalIndex() -> i32 {
        ${Sp(t)?"  return i32(globalId.x);":`  return i32((workgroupId.z * numWorkgroups.x * numWorkgroups.y +
                workgroupId.y * numWorkgroups.x + workgroupId.x) * ${r}u +
                localIndex);
        `}
      }
    `),t.pixelsOpType!=null){const I=t.pixelsOpType===to.FROM_PIXELS?`@group(0) @binding(0) var<storage, read_write> result: array<${ks(e.dtype,t.outputComponent)}>;`:`@group(0) @binding(1) var<storage, read> inBuf : array<${ks(s[0].dtype,t.outputComponent)}>;`,E=e.shape.length===3?"vec2<i32>":"i32";n.push(`
        struct Uniform {
          outShapeStrides : ${E},
          size            : i32,
          numChannels     : i32,
          alpha           : f32,
        };

        ${I}
        @group(0) @binding(2) var<uniform> uniforms: Uniform;
      `);const m=kc(t);return[Ic,n.join(`
`),ea(e.shape),t.getUserCode(),Sc(m,t)].join(`
`)}let i,o,a="struct Uniforms { NAN : f32, INFINITY : f32, ";t.variableNames.forEach((I,E)=>{const m=ht(s[E].shape.length);a+=`${I.charAt(0).toLowerCase()+I.slice(1)}Shape : ${m}, `,i=s[E].shape.length-1,o=ht(i),a+=`${I.charAt(0).toLowerCase()+I.slice(1)}ShapeStrides: ${o}, `});const l=ht(e.shape.length);a+=`outShape : ${l}, `,i=e.shape.length-1,o=ht(i),a+=`
         outShapeStrides: ${o}, `,t.size&&(a+="size : i32, "),t.uniforms&&(a+=t.uniforms),a+="};",a=M_(a),n.push(a),t.atomic?n.push(`
      @group(0) @binding(0) var<storage, read_write> result: array<atomic<i32>>;
    `):n.push(`
      @group(0) @binding(0) var<storage, read_write> result: array<${ks(e.dtype,t.outputComponent)}>;
    `),t.variableNames.forEach((I,E)=>{n.push(`
      @group(0) @binding(${1+E}) var<storage, read> ${I}: array<${t.variableComponents?ks(s[E].dtype,t.variableComponents[E]):ks(s[E].dtype,t.outputComponent)}>;
        `)}),a!==""&&n.push(`
      @group(0) @binding(${1+t.variableNames.length}) var<uniform> uniforms: Uniforms;
      `);const u=N_(e.shape,t.dispatchLayout),c=[Ic,n.join(`
`)+E_,ea(e.shape),u,D_(e.shape.length)];t.atomic||c.push(O_(e.shape,e.dtype,t.outputComponent)),t.variableNames.forEach((I,E)=>{c.push(`${ea(s[E].shape,I)}`)});const h=s.map((I,E)=>$_(I,e.shape,t.variableComponents?t.variableComponents[E]:t.outputComponent,t.dispatchLayout.x.length===e.shape.length)).join(`
`);c.push(h),c.push(t.getUserCode());const d=kc(t);return c.push(Sc(d,t)),c.join(`
`)}function T_(s,e,t){let n=s.shaderKey;if(s.pixelsOpType!=null)return n;const r=[],i=[];e.forEach(c=>{r.push(c.shape),i.push(c.dtype)}),r.push(t.shape),i.push(t.dtype);const o=e.map(c=>Di(c.shape,t.shape)),a=e.map(c=>Ht(c.shape,t.shape)).join("_"),l=o.map(c=>c.join("_")).join(";"),u=Sp(s)?"flatDispatch":"";return n+="_"+(s.workgroupSize?s.workgroupSize.join(","):"")+r.map(c=>c.length).join(",")+i.join(",")+s.variableNames.join(",")+l+a+u,n}const Ic=`
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
`,E_=`
  fn isinf(val: f32) -> bool {
    return abs(val) == uniforms.INFINITY;
  }
`;function ea(s,e=""){const t=s.length,n=e!==""?`get${e.charAt(0).toUpperCase()+e.slice(1)}CoordsFromIndex`:"getCoordsFromIndex",r=e!==""?`${e.charAt(0).toLowerCase()+e.slice(1)}ShapeStrides`:"outShapeStrides";if(t<=1)return`fn ${n}(index : i32) -> i32 { return index; }`;const i=Pt(s),o=ht(t),a=[];for(let u=0;u<t;u++)a.push(`d${u}`);if(i.length===1)return`    fn ${n}(index : i32) -> vec2<i32> {
      let d0 = index / uniforms.${r}; let d1 = index - d0 * uniforms.${r};
      return vec2<i32>(d0, d1);
    }`;let l;return l="var index2 = index;"+i.map((u,c)=>{const h=`let ${a[c]} = index2 / uniforms.${r}.${is(c)}`,d=c===i.length-1?`let ${a[c+1]} = index2 - ${a[c]} * uniforms.${r}.${is(c)}`:`index2 = index2 - ${a[c]} * uniforms.${r}.${is(c)}`;return`${h}; ${d};`}).join(""),`
    fn ${n}(index : i32) -> ${o} {
      ${l}
      return ${o}(${a.join(",")});
    }
  `}function A_(s,e){const t=s.name,n=s.shape.length,r=ht(n),i="get"+t.charAt(0).toUpperCase()+t.slice(1),o=["d0","d1","d2","d3","d4","d5"].slice(0,n),a=o.map(c=>`${c} : i32`).join(", ");if(n<1)return`
      fn ${i}() -> ${me(e)} {
        return ${me(e)}(${t}[0]);
      }
    `;const l=`uniforms.${t.charAt(0).toLowerCase()+t.slice(1)}Shape`;let u=`${n}D`;return n===0&&(u="1D"),`
    fn ${i}(${a}) -> ${me(e)} {
      return ${me(e)}(${t}[getIndexFromCoords${u}(${r}(${o.join(",")}),
        ${l})${e===1?"":` / ${e}`}]);
    }
   `}function C_(s,e,t,n){const r=s.name,i=r.charAt(0).toUpperCase()+r.slice(1),o="get"+i+"ByOutput",a=s.shape.length,l=e.length,u=ht(l);if(Ht(s.shape,e)&&n)return`
    fn ${o}Index(globalIndex : i32) -> ${me(t)} {
      return ${me(t)}(${r}[globalIndex]);
    }

    fn ${o}Coords(coords : ${u}) -> ${me(t)} {
      return ${me(t)}(${r}[${l>1?"getOutputIndexFromCoords(coords)":"coords"}${t===1?"":` / ${t}`}]);
    }
    `;const c=Di(s.shape,e),h=l-a;let d="";if(a===0)return`
    fn ${o}Index(globalIndex : i32) -> ${me(t)}{
      return get${i}();
    }

    fn ${o}Coords(coords : ${u}) -> ${me(t)}{
      return get${i}();
    }
  `;l<2&&c.length>=1?d="coords = 0;":d=c.map(m=>`coords.${is(m+h)} = 0;`).join(`
`);let w="";if(l<2&&a>0)w="coords";else if(l>1){const m=ht(a),S=s.shape.map((b,f)=>`coords.${is(f+h)}`).join(", ");w=`${m}(${S})`}else w="coords";const I=`uniforms.${r.charAt(0).toLowerCase()+r.slice(1)}Shape`,E=`${a}D`;return`
  fn ${o}Index(globalIndex : i32) -> ${me(t)} {
    var coords = getCoordsFromIndex(globalIndex);
    ${d}
    return ${me(t)}(${r}[getIndexFromCoords${E}(${w}, ${I})${t===1?"":` / ${t}`}]);
  }

  fn ${o}Coords(coordsIn : ${u}) -> ${me(t)} {
    var coords = coordsIn;
    ${d}
    return ${me(t)}(${r}[getIndexFromCoords${E}(${w}, ${I})${t===1?"":` / ${t}`}]);
  }
`}function $_(s,e,t,n){let r=A_(s,t);return s.shape.length<=e.length&&(r+=C_(s,e,t,n)),r}function N_(s,e){const{x:t,y:n=[],z:r=[]}=e,i=s.length,o=t.length+n.length+r.length;if(o!==i)return"";if(t.length===i)return`fn getOutputCoords() -> ${ht(i)}{
    let globalIndex = getGlobalIndex();
    return getCoordsFromIndex(globalIndex);
  }
  `;let a="";const l=[t,n,r];for(let d=0;d<l.length;d++){const w=l[d];if(w.length!==0)if(w.length===1)a+=`let d${w[0]} = i32(globalId[${d}]);`;else{const I=v_(w,"uniforms.outShape");a+=`var index${d} = i32(globalId[${d}]);`;for(let E=0;E<I.length;E++)a+=`let d${w[E]} = index${d} / ${I[E]};`,E===I.length-1?a+=`let d${w[E+1]} = index${d} - d${w[E]} * ${I[E]};`:a+=`index${d} = index${d} - d${w[E]} * ${I[E]};`}}const u=[];for(let d=0;d<o;d++)u.push(`d${d}`);const c=ht(o);let h=`fn getOutputCoords() -> ${c} {
  ${a}
`;return u.length===0?h+=`return ${c}(0); }`:h+=`return ${c}(${u.join(",")}); }`,h}function D_(s){let e="";switch(s){case 0:case 1:e+=`
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
        `;break;default:P(!1,()=>`Unsupported ${s}D shape`);break}return e}function Sp(s){return s.dispatch[1]===1&&s.dispatch[2]===1}function ks(s,e=1){if(s==="float32")return me(e,"f32");if(s==="int32"||s==="bool")return me(e,"i32");throw new Error(`type ${s} is not supported.`)}function O_(s,e,t){const n=s.length,r=ks(e,t);let i=`fn setOutputAtIndex(flatIndex : i32, value : ${me(t)}) {
      result[flatIndex] = ${r}(value);
    }

    fn setOutputAtIndexI32(flatIndex : i32, value : ${me(t,"i32")}) {
      result[flatIndex] = ${r}(value);
    }
    `;if(n>=2){const o=["d0","d1","d2","d3","d4","d5"].slice(0,n),a=ht(n);i+=`
      fn setOutputAtCoords(${o.map(l=>`${l} : i32`).join(", ")}, value : ${me(t)}) {
        let flatIndex = getOutputIndexFromCoords(${a}(${o.join(", ")}));
        setOutputAtIndex(flatIndex${t===1?"":` / ${t}`}, value);
      }
      fn setOutputAtCoordsI32(${o.map(l=>`${l} : i32`).join(", ")}, value : ${me(t,"i32")}) {
        let flatIndex = getOutputIndexFromCoords(${a}(${o.join(", ")}));
        setOutputAtIndexI32(flatIndex${t===1?"":` / ${t}`}, value);
      }
    `}return i}function M_(s){const e=/(\w+)\s*:\s*vec(5|6)/g;s=s.replace(e,n=>"@align(16) "+n);const t=/vec(5|6)\s*,\s*(\w+)/g;return s=s.replace(t,(n,r,i)=>`vec${r}, @align(16) ${i}`),s}function kc(s){return!(s.dispatchLayout.hasOwnProperty("y")&&s.dispatchLayout.y.length!==0||s.dispatchLayout.hasOwnProperty("z")&&s.dispatchLayout.z.length!==0)}/**
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
 */const os=s=>{let e=1;for(let t=0;t<s.length;t++)e*=s[t];return e};function Ze(s,e,t=[1,1,1],n=[1,1,1]){const[r,i,o]=[Math.ceil(os(s.x.map(a=>e[a]))/(t[0]*n[0])),s.y?Math.ceil(os(s.y.map(a=>e[a]))/(t[1]*n[1])):1,s.z?Math.ceil(os(s.z.map(a=>e[a]))/(t[2]*n[2])):1];return[r,i,o]}function P_(s,e,t,n=!1){const r=[8,8,1],i=[4,4,1];return n||(s<=8&&(i[1]=1),e<=16&&t<=16&&(r[0]=4)),{workgroupSize:r,elementsPerThread:i}}function R_(s,e,t=!1){if(t)return[8,8,1];const n=os(s.x.map(i=>e[i])),r=os(s.y.map(i=>e[i]));return n<=4?[4,16,1]:r<=4?[16,4,1]:[16,16,1]}function L_(s,e,t=!1){if(t)return[4,4,1];const n=os(s.x.map(i=>e[i])),r=os(s.y.map(i=>e[i]));return n<=4?[1,2,1]:r<=4?[2,1,1]:[2,2,1]}function Kt(s){return{x:s.map((e,t)=>t)}}function Tc(s){if(s==="float32"||s==="int32"||s==="bool"||s==="string")return 4;if(s==="complex64")return 8;throw new Error(`Unknown dtype ${s}`)}function Ip(){return!!(typeof globalThis<"u"&&globalThis.navigator&&globalThis.navigator.gpu)}var cn;(function(s){s[s.MatMulReduceProgram=0]="MatMulReduceProgram",s[s.MatMulSplitKProgram=1]="MatMulSplitKProgram",s[s.MatMulSmallOutputSizeProgram=2]="MatMulSmallOutputSizeProgram",s[s.MatMulPackedProgram=3]="MatMulPackedProgram",s[s.MatMulMax=4]="MatMulMax"})(cn||(cn={}));/**
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
 */const B_=fe().getNumber("WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD"),F_=(s,e)=>{const t=s.limits.maxComputeWorkgroupsPerDimension,n=e.dispatchLayout,r=e.dispatch;if(r.every(o=>o<=t))return r;P(r[0]>t&&n.y===void 0&&n.z===void 0,()=>"Dispatch size exceeds WebGPU limits in Y or Z dimension.");let i=Math.ceil(Math.sqrt(r[0]));return i>t?(i=Math.ceil(Math.cbrt(r[0])),P(i<=t,()=>"Total dispatch size exceeds WebGPU maximum."),[i,i,i]):[i,i,1]};class Ur extends lh{nextDataId(){return Ur.nextDataId++}constructor(e,t){if(super(),this.commandQueueOwnedIds=new WeakSet,this.dispatchCountInPass=0,this.disposed=!1,this.downloadWaitMs=0,this.tensorDataPendingDisposal=[],this.queryResolveBuffer=null,this.querySet=null,this.querySetCount=2,this.stagingPendingDisposal=[],this.uniformPendingDisposal=[],this.uploadWaitMs=0,this.hasReadSyncWarned=!1,this.hasTimestampQueryWarned=!1,!Ip())throw new Error("WebGPU is not supported on this device");this.pipelineCache={},this.device=e,this.queue=e.queue,this.commandEncoder=null,this.computePassEncoder=null,this.adapterInfo=new b_(t),this.supportTimestampQuery=this.device.features.has("timestamp-query"),this.thresholdToIncreaseWorkgroups=this.adapterInfo.intelGPUGeneration>=12?16:8,this.bufferManager=new w_(this.device),this.textureManager=new x_(this.device),this.tensorMap=new bm(this,Bo()),fe().getBool("WEBGPU_USE_PROFILE_TOOL")&&(this.dummyCanvas=document.createElement("canvas"),this.dummyCanvas.width=1,this.dummyCanvas.height=1,this.dummyContext=this.dummyCanvas.getContext("webgpu"),this.dummyContext.configure({device:e,format:"bgra8unorm"}),document.body.appendChild(this.dummyCanvas))}floatPrecision(){return 32}disposeData(e,t=!1){if(!this.tensorMap.has(e))return!0;const n=this.tensorMap.get(e);return t?n.refCount=0:n.refCount--,n.refCount>0?!1:(n.complexTensorInfos!=null&&(this.disposeData(n.complexTensorInfos.real.dataId),this.disposeData(n.complexTensorInfos.imag.dataId)),this.commandQueueOwnedIds.has(e)?(this.tensorDataPendingDisposal.push(e),!0):(this.releaseResource(e),this.tensorMap.delete(e),!0))}memory(){return{numBytesInGPU:this.bufferManager.numBytesUsed,numBytesAllocatedInGPU:this.bufferManager.numBytesAllocated,unreliable:!1}}releaseResource(e){const t=this.tensorMap.get(e);if(!(!t||!t.resource)){if(t.external){t.resource=null;return}t.resource instanceof GPUBuffer?this.bufferManager.releaseBuffer(t.resource):t.resource instanceof GPUTexture&&this.textureManager.releaseTexture(t.resource),t.resource=null}}refCount(e){return this.tensorMap.has(e)?this.tensorMap.get(e).refCount:0}incRef(e){const t=this.tensorMap.get(e);t.refCount++}decRef(e){if(this.tensorMap.has(e)){const t=this.tensorMap.get(e);t.refCount--}}write(e,t,n){if(n==="complex64"&&e!=null)throw new Error("Cannot write to a complex64 dtype. Please use tf.complex(real, imag).");const r={id:this.nextDataId()};return this.tensorMap.set(r,{dtype:n,shape:t,values:e,refCount:1}),r}move(e,t,n,r,i){if(r==="complex64")throw new Error("Cannot write to a complex64 dtype. Please use tf.complex(real, imag).");this.tensorMap.set(e,{dtype:r,shape:n,values:t,refCount:i})}submitQueue(){this.queue.submit([this.commandEncoder.finish()]),this.commandEncoder=null,this.dispatchCountInPass=0,this.commandQueueOwnedIds=new WeakSet,this.tensorDataPendingDisposal.forEach(e=>{this.releaseResource(e),this.tensorMap.delete(e)}),this.uniformPendingDisposal.forEach(e=>this.bufferManager.releaseBuffer(e)),this.stagingPendingDisposal.forEach(e=>this.bufferManager.releaseBuffer(e,!1)),this.tensorDataPendingDisposal=[],this.uniformPendingDisposal=[],this.stagingPendingDisposal=[]}ensureCommandEncoderReady(){this.commandEncoder||(this.commandEncoder=this.device.createCommandEncoder())}endComputePassEncoder(){this.computePassEncoder&&(this.computePassEncoder.end(),this.computePassEncoder=null)}async checkCompileCompletionAsync(){let e;try{e=await Promise.all(Object.values(this.pipelineCache))}catch(t){throw new Error(t.message)}Object.keys(this.pipelineCache).map((t,n)=>{this.pipelineCache[t]=e[n]})}async getBufferData(e){if(fe().getBool("WEBGPU_ENGINE_COMPILE_ONLY"))return console.warn("The data may be invalid since WEBGPU_ENGINE_COMPILE_ONLY is true, this can only be called when WEBGPU_ENGINE_COMPILE_ONLY is false"),null;const t=e.size,n=this.bufferManager.acquireBuffer(t,GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);this.ensureCommandEncoderReady(),this.endComputePassEncoder(),this.commandEncoder.copyBufferToBuffer(e,0,n,0,t),this.submitQueue(),await n.mapAsync(GPUMapMode.READ);const r=n.getMappedRange().slice(0);return n.unmap(),n!=null&&this.bufferManager.releaseBuffer(n),fe().getBool("WEBGPU_USE_PROFILE_TOOL")&&(P(this.dummyContext!==void 0,()=>"Fail to get context for profiling tool"),this.dummyContext.getCurrentTexture()),r}convertAndCacheOnCPU(e,t){const n=this.tensorMap.get(e);return n.values=t,n.values}readSync(e){const t=this.tensorMap.get(e),{values:n,complexTensorInfos:r}=t;if(n!=null||t.dtype==="string")return n;if(t.dtype==="complex64"){const E=this.readSync(r.real.dataId),m=this.readSync(r.imag.dataId),S=Do(Nu(E,m).buffer,"float32");return this.convertAndCacheOnCPU(e,S),S}this.hasReadSyncWarned||(this.hasReadSyncWarned=!0,console.warn("The performance of synchronously reading data from GPU to CPU is poor on the webgpu backend, please use asynchronous APIs instead."));const i=["opaque","premultiplied"],o=t.resource,a=o.size;P(a%4===0,()=>"Because there is 4 bytes for one pixel, buffer size must be multiple of 4.");const l=a/4,u=new ArrayBuffer(a),c=256,h=256,d=i.map(E=>new OffscreenCanvas(c,h)),w=new OffscreenCanvas(c,h);this.endComputePassEncoder(),d.map((E,m)=>{const S=E.getContext("webgpu");return S.configure({device:this.device,format:"bgra8unorm",usage:GPUTextureUsage.COPY_DST,alphaMode:i[m]}),S.getCurrentTexture()}).map((E,m)=>{const S=c*4,b=(O,$,A)=>{this.ensureCommandEncoderReady(),this.commandEncoder.copyBufferToTexture({buffer:o,bytesPerRow:S,offset:A},{texture:E},{width:O,height:$}),this.submitQueue();const g=w.getContext("2d",{willReadFrequently:!0});g.clearRect(0,0,O,$),g.drawImage(d[m],0,0);const p=g.getImageData(0,0,O,$).data,y=i[m],x=new Uint8ClampedArray(u,A,O*$*4);for(let k=0;k<x.length;k+=4)if(y==="premultiplied")x[k+3]=p[k+3];else{const C=p[k];x[k]=p[k+2],x[k+1]=p[k+1],x[k+2]=C}},f=Math.floor(l/(c*h));let _=c,v=h,T=0;for(let O=0;O<f;O++)b(_,v,T),T+=c*h*4;const N=l%(c*h);v=Math.floor(N/c),v>0&&(b(_,v,T),T+=v*(c*4)),_=N%c,_>0&&b(_,1,T)});const I=Do(u,t.dtype);return this.convertAndCacheOnCPU(e,I),I}async read(e){if(!this.tensorMap.has(e))throw new Error(`Tensor ${e} was not registered!`);const t=this.tensorMap.get(e),{values:n}=t;if(n!=null)return n;let r;if(t.dtype==="complex64"){const i=await Promise.all([this.read(t.complexTensorInfos.real.dataId),this.read(t.complexTensorInfos.imag.dataId)]),o=i[0],a=i[1];r=Nu(o,a)}else{const i=await this.getBufferData(t.resource);r=Do(i,t.dtype)}return this.convertAndCacheOnCPU(e,r),r}copyBuffer(e){const t=e.size,n=e.usage,r=this.bufferManager.acquireBuffer(t,n);return this.ensureCommandEncoderReady(),this.endComputePassEncoder(),this.commandEncoder.copyBufferToBuffer(e,0,r,0,t),this.submitQueue(),r}createTensorFromGPUData(e,t,n){let r=e.buffer;if(n==="complex64")throw new Error("Cannot write to a complex64 dtype. ");const i={id:this.nextDataId()};this.tensorMap.set(i,{dtype:n,shape:t,values:null,refCount:1,external:e.zeroCopy});const o=this.tensorMap.get(i),a=Tc(o.dtype)*he(o.shape);if(e.buffer.size<a)throw new Error(`GPUBuffer size(${e.buffer.size}) is smaller than tensor size(${a})!`);if((e.buffer.usage&(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC))!==(GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC))throw new Error("GPUBuffer.usage should include GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC!");return e.zeroCopy!==!0&&(r=this.copyBuffer(r)),o.resource=r,Bo().makeTensorFromDataId(i,t,n,this)}readToGPU(e){const t=this.tensorMap.get(e),{values:n,dtype:r,shape:i,resource:o}=t;if(r==="complex64")throw new Error("Does not support reading buffer for complex64 dtype.");if(o==null)throw n!=null?new Error("Data is not on GPU but on CPU."):new Error("There is no data on GPU or CPU.");const a=o,l=a.size,u=a.usage,c=this.bufferManager.acquireBuffer(l,u);this.ensureCommandEncoderReady(),this.endComputePassEncoder(),this.commandEncoder.copyBufferToBuffer(o,0,c,0,l),this.submitQueue();const h=this.makeTensorInfo(i,r),d=Bo().makeTensorFromTensorInfo(h),w=this.tensorMap.get(h.dataId);return w.resource=c,{tensorRef:d,buffer:c}}bufferSync(e){const t=this.readSync(e.dataId);if(e.dtype==="string")try{const n=t.map(r=>Ei(r));return Ye(e.shape,e.dtype,n)}catch{throw new Error("Failed to decode encoded string bytes into utf-8")}return Ye(e.shape,e.dtype,t)}async time(e){!this.supportTimestampQuery&&!this.hasTimestampQueryWarned&&(console.warn("This device doesn't support timestamp-query extension. Start Chrome browser with flag --enable-dawn-features=allow_unsafe_apis to try it again. Otherwise, zero will be shown for the kernel time when profiling mode is enabled."),this.hasTimestampQueryWarned=!0);const t=this.activeTimers,n=[];let r=!1;this.programTimersStack==null?(this.programTimersStack=n,r=!0):this.activeTimers.push(n),this.activeTimers=n,e();const i=as(this.activeTimers.map(u=>u.query)).filter(u=>u!=null),o=as(this.activeTimers.map(u=>u.name)).filter(u=>u!=null);this.activeTimers=t,r&&(this.programTimersStack=null);const a={uploadWaitMs:this.uploadWaitMs,downloadWaitMs:this.downloadWaitMs,kernelMs:null,wallMs:null},l=await Promise.all(i);return a.kernelMs=xm(l),a.getExtraProfileInfo=()=>l.map((u,c)=>({name:o[c],ms:u})).map(u=>`${u.name}: ${u.ms}`).join(", "),this.uploadWaitMs=0,this.downloadWaitMs=0,a}makeTensorInfo(e,t,n){return t==="string"&&n!=null&&n.length>0&&uo(n[0])&&(n=n.map(i=>Zn(i))),{dataId:this.write(n,e,t),shape:e,dtype:t}}tensorToBinding(e){if(!e)return null;const n=this.tensorMap.get(e.dataId).resource;return n instanceof GPUBuffer?{buffer:n}:n instanceof GPUTexture?n.createView():n}uploadToGPU(e){const t=this.tensorMap.get(e);if(t.resource!=null)return;const n=Tc(t.dtype)*he(t.shape);let r;const i=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;if(t.values){if(r=this.bufferManager.acquireBuffer(n,i,!0),r.mapState==="unmapped"){const o=this.bufferManager.acquireBuffer(n,GPUBufferUsage.MAP_WRITE|GPUBufferUsage.COPY_SRC,!0,!1),a=o.getMappedRange();t.dtype==="int32"||t.dtype==="bool"?new Int32Array(a).set(t.values):new Float32Array(a).set(t.values),o.unmap(),this.ensureCommandEncoderReady(),this.endComputePassEncoder(),this.commandEncoder.copyBufferToBuffer(o,0,r,0,n),this.stagingPendingDisposal.push(o)}else{const o=r.getMappedRange();t.dtype==="int32"||t.dtype==="bool"?new Int32Array(o).set(t.values):new Float32Array(o).set(t.values),r.unmap()}t.values=null}else r=this.bufferManager.acquireBuffer(n,i);t.resource=r}makeUniforms(e){let t=0,n=0;const r=[];let i=1;e.forEach(l=>{l.data.length===0&&(l.data=[1]);let u;switch(l.data.length){case 1:u=4;break;case 2:u=8;break;case 3:u=16;break;case 4:u=16;break;case 5:u=16;break;case 6:u=16;break;default:P(!1,()=>`Unsupported ${l.data.length}D shape`)}(n===5||n===6)&&(u=16),u>i&&(i=u),t=Math.ceil(t/u)*u,n=l.data.length,r.push(t),t+=l.data.length*4}),t=Math.ceil(t/i)*i;const o=new ArrayBuffer(t);e.forEach((l,u)=>{const c=r[u];l.type==="int32"?new Int32Array(o,c,l.data.length).set(l.data):l.type==="uint32"?new Uint32Array(o,c,l.data.length).set(l.data):new Float32Array(o,c,l.data.length).set(l.data)});const a=this.bufferManager.acquireBuffer(t,GPUBufferUsage.COPY_DST|GPUBufferUsage.UNIFORM);return this.queue.writeBuffer(a,0,o,0,t),this.uniformPendingDisposal.push(a),{offset:0,size:t,buffer:a}}runWebGPUProgram(e,t,n,r,i){if(i||(i=this.makeTensorInfo(e.outputShape,n)),he(i.shape)===0)return this.tensorMap.get(i.dataId).values=Cs(i.dtype,0),i;this.uploadToGPU(i.dataId),e.dispatch=F_(this.device,e);const o=t.map((l,u)=>{if(l.dtype==="complex64")throw new Error("GPGPUProgram does not support complex64 input. For complex64 dtypes, please separate the program into real and imaginary parts.");return this.uploadToGPU(l.dataId),{dtype:this.tensorMap.get(l.dataId).dtype,shape:l.shape,name:e.variableNames[u]}});e.shaderKey=T_(e,o,i);const a=fe().getBool("WEBGPU_ENGINE_COMPILE_ONLY");return e.shaderKey in this.pipelineCache||(this.pipelineCache[e.shaderKey]=S_(this.device,e,o,i,a)),e.pipeline=this.pipelineCache[e.shaderKey],a||this.recordAndSubmit(e,i,t,r),i}recordAndSubmit(e,t,n,r){if(e.pipeline instanceof Promise)throw new Error("Please call checkCompileCompletionAsync to ensure parallel compilation is done!");let i=[],o=[];const a="int32";if(e.pixelsOpType==null){i.push({type:"float32",data:[NaN]},{type:"float32",data:[1/0]}),o=n.concat(t).map(w=>w.shape);const d="int32";o.map(w=>{i.push({type:d,data:w});const I=Pt(w);i.push({type:d,data:I})})}else{const d=Pt(t.shape);i.push({type:a,data:d})}if(e.size){const d=he(e.outputShape);i.push({type:a,data:[e.outputComponent?d/e.outputComponent:d]})}r&&(i=[...i,...r]);const l=[this.tensorToBinding(t),...n.map(d=>this.tensorToBinding(d)),this.makeUniforms(i)];n.forEach(d=>{this.commandQueueOwnedIds.add(d.dataId)}),this.commandQueueOwnedIds.add(t.dataId);const u=this.device.createBindGroup({layout:e.pipeline.getBindGroupLayout(0),entries:l.map((d,w)=>({binding:w,resource:d}))}),c=this.activeTimers!=null;this.ensureCommandEncoderReady();const h={};c&&this.supportTimestampQuery?(this.endComputePassEncoder(),this.querySet==null&&(this.querySet=this.device.createQuerySet({type:"timestamp",count:this.querySetCount})),h.timestampWrites={querySet:this.querySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1},this.computePassEncoder=this.commandEncoder.beginComputePass(h)):this.computePassEncoder||(this.computePassEncoder=this.commandEncoder.beginComputePass(h)),this.computePassEncoder.setPipeline(e.pipeline),this.computePassEncoder.setBindGroup(0,u),this.computePassEncoder.dispatchWorkgroups(e.dispatch[0],e.dispatch[1],e.dispatch[2]),this.dispatchCountInPass++,(c||fe().get("WEBGPU_DEFERRED_SUBMIT_BATCH_SIZE")<=this.dispatchCountInPass||e.pixelsOpType===to.DRAW)&&(this.endComputePassEncoder(),c?this.activeTimers.push({name:e.constructor.name,query:this.getQueryTime()}):this.submitQueue())}async getQueryTime(){if(!this.supportTimestampQuery)return 0;this.queryResolveBuffer==null&&(this.queryResolveBuffer=this.bufferManager.acquireBuffer(this.querySetCount*8,GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST|GPUBufferUsage.QUERY_RESOLVE)),this.commandEncoder.resolveQuerySet(this.querySet,0,this.querySetCount,this.queryResolveBuffer,0);const e=this.bufferManager.acquireBuffer(this.querySetCount*8,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);this.commandEncoder.copyBufferToBuffer(this.queryResolveBuffer,0,e,0,this.querySetCount*8),this.submitQueue(),await e.mapAsync(GPUMapMode.READ);const t=new BigUint64Array(e.getMappedRange()),n=Number(t[1]-t[0])/1e6;return e.unmap(),this.bufferManager.releaseBuffer(e),n}shouldExecuteOnCPU(e,t=B_){return fe().getBool("WEBGPU_CPU_FORWARD")&&e.every(n=>this.tensorMap.get(n.dataId).resource==null&&he(n.shape)<t)}numDataIds(){return this.tensorMap.numDataIds()-this.tensorDataPendingDisposal.length}dispose(){this.disposed||(this.querySet!=null&&this.querySet.destroy(),this.bufferManager.dispose(),this.textureManager.dispose(),this.disposed=!0)}}Ur.nextDataId=0;/**
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
 */Ip()&&Wy("webgpu",async()=>{const s={powerPreference:fe().get("WEBGPU_USE_LOW_POWER_GPU")?"low-power":"high-performance"},e=await navigator.gpu.requestAdapter(s),t={},n=[];e.features.has("timestamp-query")&&n.push("timestamp-query"),e.features.has("bgra8unorm-storage")&&n.push(["bgra8unorm-storage"]),t.requiredFeatures=n;const r=e.limits;t.requiredLimits={maxComputeWorkgroupStorageSize:r.maxComputeWorkgroupStorageSize,maxComputeWorkgroupsPerDimension:r.maxComputeWorkgroupsPerDimension,maxStorageBufferBindingSize:r.maxStorageBufferBindingSize,maxBufferSize:r.maxBufferSize,maxComputeWorkgroupSizeX:r.maxComputeWorkgroupSizeX,maxComputeInvocationsPerWorkgroup:r.maxComputeInvocationsPerWorkgroup};const i=await e.requestDevice(t),o=await e.requestAdapterInfo();return new Ur(i,o)},3);/**
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
 */class U_{constructor(e,t,n){this.uniforms="",this.variableNames=["x"],this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=t.map((r,i)=>r[0]+e[i]+r[1]),this.dispatchLayout=Kt(this.outputShape),this.dispatch=Ze(this.dispatchLayout,this.outputShape,this.workgroupSize),this.xShape=e,t.map((r,i)=>{this.uniforms+=` pad${i} : vec2<i32>,`}),this.offset=n==="reflect"?0:1,this.shaderKey=`mirrorPad_${n}`}getUserCode(){const e=this.xShape.length,t=this.xShape.map((u,c)=>`uniforms.pad${c}[0]`).join(","),n=this.xShape.map((u,c)=>`uniforms.pad${c}[0] + uniforms.xShape${e>1?`[${c}]`:""}`).join(","),r=e===1?"start":"start[i]",i=e===1?"end":"end[i]",o=e===1?"outC":"outC[i]",a=ht(e),l=e>1?["coords[0]","coords[1]","coords[2]","coords[3]"].slice(0,e):"coords";return`
      ${je("index")} {
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
 */const z_={kernelName:Ig,backendName:"webgpu",kernelFunc:({inputs:s,attrs:e,backend:t})=>{const{x:n}=s,{paddings:r,mode:i}=e,o=t,a=r.map(c=>({type:"int32",data:[c[0],c[1]]})),l=new U_(n.shape,r,i);return o.runWebGPUProgram(l,[n],n.dtype,a)}};/**
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
 */function Wn(s){const{inputs:e}=s,{x:t}=e;return s.backend.incRef(t.dataId),{dataId:t.dataId,shape:t.shape,dtype:t.dtype}}const V_={kernelName:Ga,backendName:"webgpu",kernelFunc:Wn};/**
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
 */function G_(s,e=!1){const t=s.length,n=ht(t),r=s.map((h,d)=>`uniforms.pad${d}[0]`).join(","),i=s.map((h,d)=>`uniforms.pad${d}[0] + uniforms.xShape${t>1?`[${d}]`:""}`).join(","),o=t>1?`${n}(${r})`:`${r}`,a=t>1?`${n}(${i})`:`${i}`,l=t>1?"any(paddedCoords < start)":"paddedCoords < start",u=t>1?"any(paddedCoords >= end)":"paddedCoords >= end",c=t>1?["coords[0]","coords[1]","coords[2]","coords[3]"].slice(0,t):"coords";return`
        let start = ${o};
        let end = ${a};
        if (${l} || ${u}) {
          setOutputAtIndex(index, ${e?0:"uniforms.constantValue"});
        } else {
          let coords = paddedCoords - start;
          setOutputAtIndex(index, getX(${c}));
        }
  `}class W_{constructor(e,t){this.variableNames=["x"],this.uniforms="constantValue : f32,",this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=t.map((n,r)=>n[0]+e[r]+n[1]),this.dispatchLayout=Kt(this.outputShape),this.dispatch=Ze(this.dispatchLayout,this.outputShape,this.workgroupSize),t.map((n,r)=>{this.uniforms+=` pad${r} : vec2<i32>,`}),this.xShape=e,this.shaderKey="pad"}getUserCode(){return`
      ${je("index")} {
        if (index < uniforms.size) {
          let paddedCoords = getCoordsFromIndex(index);
          ${G_(this.xShape)}
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
 */class q_{constructor(e){this.variableNames=[],this.outputShape=[],this.uniforms="value : f32,",this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=Kt(this.outputShape),this.dispatch=Ze(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey="fill"}getUserCode(){return`
    ${je("index")} {
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
 */function kp(s){const{backend:e,attrs:t}=s,{shape:n,value:r}=t;let{dtype:i}=t;if(i=i||Cr(r),i==="string"){const o=He(i,he(n));return o.fill(r),e.makeTensorInfo(n,i,o)}else{const o=new q_(n),a=[{type:"float32",data:[r]}];return e.runWebGPUProgram(o,[],i,a)}}/**
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
 */const H_=s=>{const{inputs:e,backend:t,attrs:n}=s,{x:r}=e,{paddings:i,constantValue:o}=n;if(i.every(u=>Ht(u,[0,0])))return Wn({inputs:{x:r},backend:t});if(he(r.shape)===0){const u=i.map((c,h)=>c[0]+r.shape[h]+c[1]);return kp({backend:t,attrs:{shape:u,value:o,dtype:r.dtype}})}const a=[{type:"float32",data:[o]}];i.map(u=>a.push({type:"int32",data:[u[0],u[1]]}));const l=new W_(r.shape,i);return t.runWebGPUProgram(l,[r],r.dtype,a)},j_={kernelName:bh,backendName:"webgpu",kernelFunc:H_};/**
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
 */function K_(s){const e=new Float32Array(s.length);for(let t=0;t<s.length;++t)e[t]=Math.abs(s[t]);return e}/**
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
 */function wt(s){return(e,t,n,r,i)=>{const o=St(e,t),a=o.length,l=Pt(o),u=he(o),c=Cs(i,u),h=e.length,d=t.length,w=Pt(e),I=Pt(t),E=Di(e,o),m=Di(t,o);if(E.length+m.length===0)for(let S=0;S<c.length;++S)c[S]=s(n[S%n.length],r[S%r.length]);else for(let S=0;S<c.length;++S){const b=Ua(S,a,l),f=b.slice(-h);E.forEach(N=>f[N]=0);const _=la(f,h,w),v=b.slice(-d);m.forEach(N=>v[N]=0);const T=la(v,d,I);c[S]=s(n[_],r[T])}return[c,o]}}/**
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
 */function X_(s,e,t,n){if(n==="int32"){const r=Int32Array.from(s);return[e,"int32",r]}if(n==="bool"){const r=ho([0],t),[i,o]=wt((a,l)=>a!==l?1:0)(e,[],s,r,"bool");return[o,"bool",i]}throw new Error(`Error in Cast: failed to cast ${t} to ${n}`)}/**
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
 */const Y_=wt(((s,e)=>s+e));/**
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
 */function Q_(s,e,t,n,r){const i=he(n),o=Gn(r,t);for(let a=0;a<s.length;a++){const l=s[a];if(l<0)throw new Error("Input x must be non-negative!");l>=r||(i>0?o[l]+=e[a]:o[l]+=1)}return o}function Z_(s,e,t,n=!1){const r=s.shape[0],i=s.shape[1],o=Ye([r,t],e.dtype);for(let a=0;a<r;a++)for(let l=0;l<i;l++){const u=s.get(a,l);if(u<0)throw new Error("Input x must be non-negative!");u>=t||(n?o.set(1,a,u):e.size>0?o.set(o.get(a,u)+e.get(a,l),a,u):o.set(o.get(a,u)+1,a,u))}return o}/**
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
 */const J_=wt(((s,e)=>s&e));/**
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
 */function Dn(s){return(e,t,n)=>{const r=He(t,e.length);for(let i=0;i<e.length;++i)r[i]=s(e[i],n);return r}}/**
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
 */const e3=Dn(s=>Math.ceil(s));/**
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
 */function t3(s,e,t,n){const r=He(t,he(e));if(n&&t!=="string"){let i=0;s.forEach(o=>{const a=he(o.shape);r.set(o.vals,i),i+=a})}else{let i=0;s.forEach(o=>{const a=t==="string"?pf(o.vals):o.vals;let l=0;for(let u=0;u<o.shape[0];++u){const c=u*e[1]+i;for(let h=0;h<o.shape[1];++h)r[c+h]=a[l++]}i+=o.shape[1]})}return r}/**
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
 */const n3=wt((s,e)=>s===e?1:0);/**
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
 */const s3=Dn(s=>Math.exp(s));/**
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
 */const r3=Dn(s=>Math.expm1(s));/**
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
 */const i3=Dn(s=>Math.floor(s));/**
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
 */const o3=wt((s,e)=>Math.floor(s/e));/**
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
 */function a3(s,e,t,n,r,i,o,a,l){const u=Ye([n,i],t);for(let c=0;c<n;c++){const h=[];let d=0;for(let w=0;w<r;w++){const I=s[c*r+w];d+=I*o[w],h.push(I)}if(d<0||d>=l/i)throw new Error(`Invalid indices: ${h} does not index into ${a}`);for(let w=0;w<i;w++)u.values[c*i+w]=e.get(...e.indexToLoc(d*i+w))}return u}/**
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
 */function l3(s,e,t){const n=Ye(t,s.dtype);for(let r=0;r<n.size;++r){const o=n.indexToLoc(r).slice(),a=o[0],l=o[2],u=e.locToIndex([a,l]);o[2]=e.values[u];const c=s.locToIndex(o);0<=c&&c<s.values.length&&(n.values[r]=s.values[c])}return n}/**
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
 */const u3=wt((s,e)=>s>e?1:0);/**
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
 */const c3=wt((s,e)=>s>=e?1:0);/**
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
 */const h3=wt((s,e)=>s<e?1:0);/**
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
 */const f3=wt((s,e)=>s<=e?1:0);/**
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
 */function d3(s,e,t){const n=(e-s)/(t-1),r=Gn(t,"float32");r[0]=s;for(let i=1;i<r.length;i++)r[i]=r[i-1]+n;return r}/**
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
 */const p3=Dn(s=>Math.log(s));/**
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
 */function m3(s,e,t,n){const r=Cs(n,he(t));for(let i=0;i<r.length;++i){const o=i*e;let a=s[o];for(let l=0;l<e;++l){const u=s[o+l];(Number.isNaN(u)||u>a)&&(a=u)}r[i]=a}return r}/**
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
 */const g3=wt(((s,e)=>Math.max(s,e)));/**
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
 */const y3=wt(((s,e)=>Math.min(s,e)));/**
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
 */const Tp=wt(((s,e)=>s*e));/**
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
 */function b3(s,e,t){const n=py(-1,t);return Tp([],e,n,s,t)}/**
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
 */const w3=wt(((s,e)=>s!==e?1:0));/**
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
 */function x3(s,e,t,n,r){const i=e.length,o=he(e),a=Pt(e),l=Pt(r),u=Cs(t,he(r));for(let c=0;c<o;++c){const h=Ua(c,i,a),d=new Array(h.length);for(let I=0;I<d.length;I++)d[I]=h[n[I]];const w=la(d,i,l);u[w]=s[c]}return u}/**
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
 */function v3(s,e,t,n){const[r,i]=Qa(s,n),o=Wa(e,"int32"),a=Gn(he(r),o),l=he(i);for(let u=0;u<a.length;++u){const c=u*l;let h=1;for(let d=0;d<l;++d)h*=t[c+d];a[u]=h}return{outVals:a,outShape:r,outDtype:o}}/**
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
 */function _3(s,e,t){s.forEach((n,r)=>{if(n<0||n>=t){const i=Ua(r,e.length,Pt(e)).join(",");throw new Error(`indices[${i}] = ${n} is not in [0, ${t})`)}})}function S3(s,e){for(let t=0;t<s.length;++t){const n=s[t],r=t===s.length-1?e:s[t+1].length;if(n.length===0)throw new Error("Ragged splits may not be empty");if(n[0]<0)throw new Error("Ragged splits must be non-negative");if(n[n.length-1]>r)throw new Error("Ragged splits must not point past values");for(let i=1;i<n.length;++i)if(n[i-1]>n[i])throw new Error("Ragged splits must be sorted in ascending order")}}function I3(s,e,t,n){const r=[];let i=0;const o=e.length-1+t.length,a=new Array(o).fill(null).map(()=>[0]);S3(t,n);let l=1;for(let u=0;u<e.length-1;++u){l*=e[u];const c=e[u+1];for(let h=1;h<l+1;++h)a[u].push(h*c)}for(let u=0;u<s.length;++u){let c=s[u],h=s[u]+1;for(let d=0;d<t.length;++d){const w=t[d],I=d+e.length-1;if(I>=0){const E=a[I],m=E[E.length-1]-w[c];for(let S=c;S<h;++S)a[I].push(w[S+1]+m)}c=w[c],h=w[h]}h!==c&&(r.push([c,h]),i+=h-c)}return{outSplits:a,valueSlices:r,numValues:i}}function k3(s){const e=[];for(let t=0;t<s.length;++t){const n=s[t].length,r=He("int32",n);e.push(r),s[t].forEach((i,o)=>r[o]=i)}return e}function Ec(s,e){const t=s.slice(0,e);for(;t.length<e;)t.push(1);for(let n=e;n<s.length;n++)t[e-1]*=s[n];return t}function T3(s,e,t,n,r,i){const o=Ec(e,2)[1],a=Ec(i,2)[1];let l=0;for(const u of t)for(let c=u[0];c<u[1];++c){for(let h=0;h<n;++h)r[l*a+h]=s[c*o+h];++l}}function E3(s,e,t,n,r){const i=e.slice();i[0]=r;const o=He(t,he(i)),a=s.length,l=a===0?0:a/e[0];return T3(s,e,n,l,o,i),[o,i]}function A3(s,e,t,n,r,i,o,a){if(s.length===0)throw new Error("paramsNestedSplits must be non empty");if(e[0].length===0)throw new Error("Split tensors must not be scalars");const l=e[0][0]-1;if(_3(i,o,l),n.length===0)throw new Error("params.rank must be nonzero");const u=n[0],{outSplits:c,valueSlices:h,numValues:d}=I3(i,o,s,u),w=k3(c),I=E3(t,n,r,h,d);return[w,I[0],I[1]]}/**
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
 */const Ac=2147483647;function C3(s,e,t,n,r,i,o){if(e.length>1)throw new Error("starts must be a scalar or vector");if(r.length>1)throw new Error("limits must be a scalar or vector");if(o.length>1)throw new Error("deltas must be a scalar or vector");const a=e.length===0,l=r.length===0,u=o.length===0,c=[];a||c.push(e[0]),l||c.push(r[0]),u||c.push(o[0]);for(let m=1;m<c.length;++m)if(c[m]!==c[m-1])throw new Error("starts, limits, and deltas must have the same shape");const h=c.length===0?1:c[0],d=He("int32",h+1);d[0]=0;for(let m=0;m<h;++m){const S=a?s[0]:s[m],b=l?n[0]:n[m],f=u?i[0]:i[m];if(f===0)throw new Error("Requires delta != 0");let _;if(f>0&&b<S||f<0&&b>S)_=0;else if(_=Math.ceil(Math.abs((b-S)/f)),_>Ac)throw new Error(`Requires ((limit - start) / delta) <= ${Ac}`);d[m+1]=d[m]+_}const w=d[h],I=He(t,w);let E=0;for(let m=0;m<h;++m){const S=d[m+1]-d[m];let b=a?s[0]:s[m];const f=u?i[0]:i[m];for(let _=0;_<S;++_)I[E++]=b,b+=f}return[d,I]}/**
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
 */var Ft=pn;class no{constructor(e,t,n,r,i,o,a,l,u,c){this.shape=e,this.shapeShape=t,this.values=n,this.valuesShape=r,this.valuesDType=i,this.defaultValue=o,this.defaultValueShape=a,this.rowPartitionValues=l,this.rowPartitionValuesShapes=u,this.rowPartitionTypes=cx(c),this.raggedRank=hx(this.rowPartitionTypes)}getRowPartitionTypeByDimension(e){return this.rowPartitionTypes[0]===Ft.FIRST_DIM_SIZE?this.rowPartitionTypes[e+1]:this.rowPartitionTypes[e]}getRowPartitionTensor(e){return this.rowPartitionTypes[0]===Ft.FIRST_DIM_SIZE?this.rowPartitionValues[e+1]:this.rowPartitionValues[e]}getMaxWidth(e){const t=this.getRowPartitionTensor(e-1);switch(this.getRowPartitionTypeByDimension(e-1)){case Ft.VALUE_ROWIDS:return no.getMaxWidthValueRowID(t);case Ft.ROW_SPLITS:return no.getMaxWidthRowSplit(t);default:throw new Error(`Cannot handle partition type ${Ft[this.getRowPartitionTypeByDimension(e-1)]}`)}}static getMaxWidthRowSplit(e){const t=e.length;if(t===0||t===1)return 0;let n=0;for(let r=0;r<t-1;++r){const i=e[r+1]-e[r];i>n&&(n=i)}return n}static getMaxWidthValueRowID(e){const t=e.length;if(t===0)return 0;let n=0,r=e[0],i=0;for(let o=1;o<t;++o){const a=e[o];a!==r&&(r=a,i=Math.max(o-n,i),n=o)}return Math.max(t-n,i)}tensorShapeFromTensor(e,t,n=!0){if(t.length===0){if(e[0]===-1)return[];throw new Error("The only valid scalar shape tensor is the fully unknown shape specified as -1.")}return $c(e,n)}calculateOutputSize(e){const t=this.valuesShape,n=this.defaultValueShape;fx(n,t);const r=this.tensorShapeFromTensor(this.shape,this.shapeShape),o=ux(this.raggedRank,r,t);o[0]<0&&(o[0]=e);for(let a=1;a<=this.raggedRank;++a)o[a]<0&&(o[a]=this.getMaxWidth(a));return o}calculateFirstParentOutputIndex(e,t,n){const r=Math.min(e,n),i=[];let o=0;for(let a=0;a<r;++a,o+=t)i.push(o);for(let a=r;a<e;++a)i.push(-1);return P(i.length===e,()=>"Final length of result must be equal to firstDimension."),i}calculateOutputIndexRowSplit(e,t,n,r){const i=e.length,o=[];for(let a=0;a<i-1;++a){const l=e[a+1]-e[a];let u=Math.min(r,l),c=t[a];c===-1&&(u=0);for(let h=0;h<u;++h)o.push(c),c+=n;for(let h=0;h<l-u;++h)o.push(-1)}if(i>0&&o.length!==e[i-1])throw new Error("Invalid row split size.");return o}calculateOutputIndexValueRowID(e,t,n,r){const i=e.length,o=[];if(i===0)return[];let a=0,l=e[0];if(l>=t.length)throw new Error(`Got currentValueRowId=${l}, which is not less than ${t.length}`);let u=t[l];o.push(u);for(let c=1;c<i;++c){const h=e[c];if(h===l)u>=0&&(++a,a<r?u+=n:u=-1);else{if(a=0,l=h,h>=t.length)throw new Error(`Got nextValueRowId=${h} which is not less than ${t.length}`);u=t[h]}o.push(u)}if(o.length!==e.length)throw new Error("Invalid row ids.");return o}calculateOutputIndex(e,t,n,r){const i=this.getRowPartitionTensor(e),o=this.getRowPartitionTypeByDimension(e);switch(o){case Ft.VALUE_ROWIDS:return this.calculateOutputIndexValueRowID(i,t,n,r);case Ft.ROW_SPLITS:if(i.length-1>t.length)throw new Error(`Row partition size is greater than output size: ${i.length-1} > ${t.length}`);return this.calculateOutputIndexRowSplit(i,t,n,r);default:throw new Error(`Unsupported partition type: ${Ft[o]}`)}}getFirstDimensionSize(){const e=this.rowPartitionValues[0];if(this.rowPartitionTypes.length===0)throw new Error("No row_partition_types given.");const t=this.rowPartitionTypes[0];switch(t){case Ft.FIRST_DIM_SIZE:return e[0];case Ft.VALUE_ROWIDS:throw new Error("Cannot handle VALUE_ROWIDS in first dimension.");case Ft.ROW_SPLITS:return this.rowPartitionValuesShapes[0][0]-1;default:throw new Error(`Cannot handle type ${Ft[t]}`)}}compute(){if(this.rowPartitionValues[0].length<=0)throw new Error("Invalid first partition input. Tensor requires at least one element.");const t=this.getFirstDimensionSize(),n=this.calculateOutputSize(t),r=new Array(this.raggedRank+1);r[r.length-1]=1;for(let l=r.length-2;l>=0;--l)r[l]=r[l+1]*n[l+1];const i=$c(n,!1),o=He(this.valuesDType,he(i));if(r[0]*n[0]>0){let l=this.calculateFirstParentOutputIndex(t,r[0],n[0]);for(let u=1;u<=this.raggedRank;++u)l=this.calculateOutputIndex(u-1,l,r[u],n[u]);this.setOutput(this.raggedRank,l,o,i)}return[i,o]}setOutput(e,t,n,r){if(n.length===0)return;const i=this.values,o=n;let a=r.slice();a=a.slice(e+1);const l=he(a),u=t.length;let c=this.defaultValue;if(c.length!==l&&c.length!==1){const I=this.defaultValueShape;Y(()=>{const E=se(c,I);c=di(E,a).dataSync()})}let h=0,d=0,w=0;for(let I=0;I<=u;++I){let E=I<u?t[I]:-1;if(E===w){++w;continue}if(d<w){const m=i.subarray(h*l),S=o.subarray(d*l),b=(w-d)*l;Cc(S,m,b)}if(I>=u){const m=n.length;E=Math.floor(m/l)}if(E>w)if(this.defaultValue.length===1)o.subarray(w*l,E*l).fill(this.defaultValue[0]),w=E;else for(;E>w;){const m=o.slice(w*l);Cc(m,c,l),++w}E<0?(h=I+1,d=w):(h=I,d=w,w=d+1)}}}function Cc(s,e,t){for(let n=0;n<t;n++)s[n]=e[n]}function $c(s,e){const t=[];for(let n of s){if(n<0){if(!e)throw new Error(`Dimension ${n} must be >= 0`);if(n<-1)throw new Error(`Dimension ${n} must be >= -1`);n=-1}t.push(n)}return t}function $3(s,e,t,n,r,i,o,a,l,u){return new no(s,e,t,n,r,i,o,a,l,u).compute()}/**
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
 */function N3(s,e,t,n){const r=s===e,i=s<e&&t<0,o=e<s&&t>1;if(r||i||o)return Gn(0,n);const a=Math.abs(Math.ceil((e-s)/t)),l=Gn(a,n);e<s&&t===1&&(t=-1),l[0]=s;for(let u=1;u<l.length;u++)l[u]=l[u-1]+t;return l}/**
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
 */const D3=Dn(s=>1/Math.sqrt(s));/**
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
 */function O3(s,e,t,n,r,i,o,a,l,u){const c=[n/r,r],h=s.values,d=e.values;if(n===0)return Ye(t,e.dtype);const w=l instanceof Ai?l:Ye(c,e.dtype);typeof l=="string"||typeof l=="number"?w.values.fill(l):typeof l=="boolean"&&w.values.fill(+l);for(let I=0;I<i;I++){const E=[];let m=0;for(let S=0;S<o;S++){const b=h[I*o+S];E.push(b),m+=b*a[S]}if(m<0||m>=n/r)throw new Error(`Invalid indices: ${E} does not index into ${t}`);for(let S=0;S<r;S++)u?w.values[m*r+S]+=d[I*r+S]:w.values[m*r+S]=e.rank===0?d[0]:d[I*r+S]}return w}/**
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
 */const M3=Dn(s=>1/(1+Math.exp(-s)));/**
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
 */function P3(s,e,t,n,r){const i=nx(n,e,t),o=he(t),a=Pt(n);if(i){const h=sx(e,a);return r==="string"?s.slice(h,h+o):s.subarray(h,h+o)}const l=r==="string"?pf(s):s,u=Ye(n,r,l),c=Ye(t,r);for(let h=0;h<c.size;++h){const d=c.indexToLoc(h),w=d.map((I,E)=>I+e[E]);c.set(u.get(...w),...d)}return r==="string"?Dx(c.values):c.values}/**
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
 */function R3(s,e,t,n,r,i,o){const a=e[0],l=i[0],u=new Array(l),c=new Array(a),h=e[1];if(l===0){if(a!==0)throw new Error(vx(a));const m=He(t,0),S=He(r,0);return[m,[0,h],S,u,c]}let d=!0,w=0;const I=new Array(l).fill(0);for(let m=0;m<a;++m){const S=s[m*h];if(S<0)throw new Error(_x(m,S));if(S>=l)throw new Error(Sx(m,S,l));++I[S],d=d&&S>=w,w=S}let E=!0;for(let m=0;m<l;++m){const S=I[m]===0;u[m]=S,E=E&&!S,I[m]=Math.max(I[m],1),m>0&&(I[m]+=I[m-1])}if(E&&d){const m=s,S=n;for(let b=0;b<a;++b)c[b]=b;return[m,[a,h],S,u,c]}else{const m=I[l-1],S=He(t,m*h),b=He(r,m),f=new Array(l).fill(0);for(let _=0;_<a;++_){const v=s[_*h],T=f[v],N=(v===0?0:I[v-1])+T;f[v]++;for(let O=0;O<h;++O)S[N*h+O]=s[_*h+O];b[N]=n[_],c[_]=N}for(let _=0;_<l;++_)if(f[_]===0){const T=_===0?0:I[_-1];S[T*h+0]=_;for(let N=1;N<h;++N)S[T*h+N]=0;b[T]=o}return[S,[m,h],b,u,c]}}/**
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
 */function L3(s,e,t,n,r){const i=he(n),o=e[0],a=r.length,l=[];let u=1,c=-1;for(let m=0;m<a;++m){const S=r[m];if(S===-1){if(c!==-1)throw new Error(Ix(c,m));c=m,l.push(1)}else{if(S<0)throw new Error(kx(m,S));u*=S,l.push(S)}}if(c!==-1){if(u<=0)throw new Error(Tx());const m=Math.trunc(i/u);if(u*m!==i)throw new Error(Ex(n,l));l[c]=m}if(he(l)!==i)throw new Error(Ax(n,l));const d=n.length,w=[];if(d>0){w[d-1]=1;for(let m=d-2;m>=0;--m)w[m]=w[m+1]*n[m+1]}const I=[];if(a>0){I[a-1]=1;for(let m=a-2;m>=0;--m)I[m]=I[m+1]*l[m+1]}const E=He(t,o*a);for(let m=0;m<o;++m){let S=0;for(let b=0;b<d;++b)S+=s[m*d+b]*w[b];for(let b=0;b<a;++b)E[m*a+b]=Math.trunc(S/I[b]),S%=I[b]}return[E,[o,a],l]}/**
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
 */function B3(s,e,t,n,r,i=!1,o=0){const a=n.length,l=[e[0],s.length/e[0]],u=l[1],h=a>0?r[a-1]+1:0;if(h<0)throw new Error(Du());const d=e.slice();d[0]=h;const w=d.reduce((f,_)=>f*_,1),I=He(t,w);if(a===0)return h>0&&I.fill(o),[I,d];if(h<=0)throw new Error(Du());let E=0,m=1,S=0,b=r[E];for(;;){let f=0;if(m<a){if(f=r[m],b===f){++m;continue}if(b>=f)throw new Error(Cx())}if(b<0||b>=h)throw new Error($x(b,h));b>S&&I.fill(o,S*u,b*u);for(let _=E;_<m;++_){const v=n[_];if(v<0||v>=l[0])throw new Error(Nx(_,n[_],l[0]));for(let T=0;T<u;T++)I[b*u+T]+=s[v*u+T]}if(i)for(let _=0;_<u;_++)I[b*u+_]/=m-E;if(E=m,++m,S=b+1,b=f,m>a)break}return S<h&&I.fill(o,S*u,h*u),[I,d]}/**
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
 */const F3=Dn(s=>Math.sqrt(s));/**
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
 */const U3=wt(((s,e)=>{const t=s-e;return t*t}));/**
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
 */const z3=Dn((s,e)=>{const{pattern:t,replaceGlobal:n,rewrite:r}=e;return s.replace(new RegExp(t,n?"g":""),r)});/**
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
 */function V3(s,e,t,n){const r=Ye(s,e.dtype);for(let i=0;i<r.size;i++){const o=r.indexToLoc(i),a=new Array(o.length);for(let l=0;l<a.length;l++)a[l]=o[l]*t[l]+n[l];r.set(e.get(...a),...o)}return r}/**
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
 */class G3{constructor(e,t,n,r,i,o){this.separator=Zn(e),this.nGramWidths=t,this.leftPad=Zn(n),this.rightPad=Zn(r),this.padWidth=i,this.preserveShort=o}getPadWidth(e){return Math.min(this.padWidth<0?e-1:this.padWidth,e-1)}getNumNGrams(e,t){const n=this.getPadWidth(t);return Math.max(0,e+2*n-t+1)}createNGrams(e,t,n,r,i,o){for(let a=0;a<i;++a){const l=this.getPadWidth(o),u=Math.max(0,l-a),c=Math.max(0,l-(i-(a+1))),h=o-(u+c),d=t+(u>0?0:a-l);let w=0;w+=u*this.leftPad.length;for(let b=0;b<h;++b)w+=e[d+b].length;w+=c*this.rightPad.length;const I=u+c+h-1;w+=I*this.separator.length,n[r+a]=new Uint8Array(w);const E=n[r+a];let m=0;const S=b=>b.forEach(f=>E[m++]=f);for(let b=0;b<u;++b)S(this.leftPad),S(this.separator);for(let b=0;b<h-1;++b)S(e[d+b]),S(this.separator);if(h>0){S(e[d+h-1]);for(let b=0;b<c;++b)S(this.separator),S(this.rightPad)}else{for(let b=0;b<c-1;++b)S(this.rightPad),S(this.separator);S(this.rightPad)}}}compute(e,t){const n=e.length,r=t.length;if(r>0){let l=t[0];if(l!==0)throw new Error(`First split value must be 0, got ${l}`);for(let u=1;u<r;++u){let c=t[u]>=l;if(c=c&&t[u]<=n,!c)throw new Error(`Invalid split value ${t[u]}, must be in [${l}, ${n}]`);l=t[u]}if(l!==n)throw new Error(`Last split value must be data size. Expected ${n}, got ${l}`)}const i=r-1,o=He("int32",r);if(n===0||r===0){const l=new Array(n);for(let u=0;u<=i;++u)o[u]=0;return[l,o]}o[0]=0;for(let l=1;l<=i;++l){const u=t[l]-t[l-1];let c=0;this.nGramWidths.forEach(h=>{c+=this.getNumNGrams(u,h)}),this.preserveShort&&u>0&&c===0&&(c=1),o[l]=o[l-1]+c}const a=new Array(o[i]);for(let l=0;l<i;++l){const u=t[l];let c=o[l];if(this.nGramWidths.forEach(h=>{const d=t[l+1]-t[l],w=this.getNumNGrams(d,h);this.createNGrams(e,u,a,c,w,h),c+=w}),this.preserveShort&&c===o[l]){const h=t[l+1]-t[l];if(h===0)continue;const d=h+2*this.padWidth;this.createNGrams(e,u,a,c,1,d)}}return[a,o]}}function W3(s,e,t,n,r,i,o,a){return new G3(t,n,r,i,o,a).compute(s,e)}/**
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
 */function q3(s,e,t,n){if(!s.length)return;if(e.length===0){for(let i=0;i<s.length;++i)n.push(s.subarray(i,i+1));return}if(e.length===1){const i=e[0];let o=s.indexOf(i);for(;o!==-1;){const a=s.subarray(0,o);(!t||a.length!==0)&&n.push(a),s=s.subarray(o+1),o=s.indexOf(i)}(!t||s.length!==0)&&n.push(s);return}let r=0;for(let i=0;i<s.length+1;i++)if(i===s.length||e.indexOf(s[i])!==-1){const o=s.subarray(r,i);(!t||o.length!==0)&&n.push(o),r=i+1}}function H3(s,e,t){const n=s.length,r=[];let i=0,o=0;const a=new Array(n);for(let d=0;d<n;++d){const w=r.length;q3(s[d],e,t,r);const I=r.length-w;a[d]=I,i+=I,o=Math.max(o,I)}const l=He("int32",i*2),u=new Array(i),c=[n,o];let h=0;for(let d=0;d<n;++d)for(let w=0;w<a[d];++w)l[h*2]=d,l[h*2+1]=w,u[h]=r[h],++h;return[l,u,c]}/**
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
 */function j3(s,e){const t=He("int32",s.length);for(let n=0;n<s.length;++n)t[n]=dy(s[n]).modulo(e).getLowBitsUnsigned();return t}/**
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
 */const K3=wt(((s,e)=>s-e));/**
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
 */function X3(s,e){const t=new Array(s.rank);for(let r=0;r<t.length;r++)t[r]=s.shape[r]*e[r];const n=Ye(t,s.dtype);for(let r=0;r<n.values.length;++r){const i=n.indexToLoc(r),o=new Array(s.rank);for(let l=0;l<o.length;l++)o[l]=i[l]%s.shape[l];const a=s.locToIndex(o);n.values[r]=s.values[a]}return n}/**
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
 */const or=(s,e)=>{const t=e.value-s.value;return t===0?s.index-e.index:t};function Ep(s,e,t=0,n=s.length-1){for(;n>t;){if(n-t>600){const a=n-t+1,l=e-t+1,u=Math.log(a),c=.5*Math.exp(2*u/3),h=.5*Math.sqrt(u*c*(a-c)/a)*Math.sign(l-a/2),d=Math.max(t,Math.floor(e-l*c/a+h)),w=Math.min(n,Math.floor(e+(a-l)*c/a+h));Ep(s,e,d,w)}const r=s[e];let i=t,o=n;for(vs(s,t,e),or(s[n],r)>0&&vs(s,t,n);i<o;){for(vs(s,i,o),i++,o--;or(s[i],r)<0;)i=i+1;for(;or(s[o],r)>0;)o=o-1}or(s[t],r)===0?vs(s,t,o):(o=o+1,vs(s,o,n)),o<=e&&(t=o+1),e<=o&&(n=o-1)}}function Y3(s,e,t,n,r){const i=e[e.length-1],[o,a]=[s.length/i,i],l=Cs(t,o*n),u=Cs("int32",o*n);for(let h=0;h<o;h++){const d=h*a,w=s.subarray(d,d+a);let I=new Array(w.length);w.forEach((b,f)=>I[f]={value:b,index:f}),n<I.length&&(Ep(I,n),I=I.slice(0,n)),r&&I.sort(or);const E=h*n,m=l.subarray(E,E+n),S=u.subarray(E,E+n);for(let b=0;b<n;b++)m[b]=I[b].value,S[b]=I[b].index}const c=e.slice();return c[c.length-1]=n,[Ye(c,t,l),Ye(c,"int32",u)]}/**
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
 */function Q3(s,e,t,n){const r=Ar(e,t)[0],i=[1,t[0],1];for(let I=0;I<r;I++)i[0]*=t[I];i[1]=t[r];for(let I=r+1;I<t.length;I++)i[2]*=t[I];const o=new Map,a=new Int32Array(t[r]),l=new Ai(i,n,s),u=[],c=i[0]===1&&i[2]===1;for(let I=0;I<t[r];I++){let E;if(c)E=s[I].toString();else{const S=[];for(let b=0;b<i[0];b++)for(let f=0;f<i[2];f++)S.push(l.get(b,I,f));E=S.join(",")}const m=o.get(E);if(m!=null)a[I]=m;else{const S=o.size;o.set(E,S),a[I]=S,u.push(I)}}const h=i.slice();h[1]=o.size;const d=new Ai(h,n);u.forEach((I,E)=>{for(let m=0;m<i[0];m++)for(let S=0;S<i[2];S++)d.set(l.get(m,I,S),m,E,S)});const w=t.slice();return w[r]=h[1],{outputValues:d.values,outputShape:w,indices:a}}/**
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
 */var Z3=Object.freeze({__proto__:null,addImpl:Y_,bincountImpl:Q_,bincountReduceImpl:Z_,bitwiseAndImpl:J_,castImpl:X_,ceilImpl:e3,concatImpl:t3,equalImpl:n3,expImpl:s3,expm1Impl:r3,floorDivImpl:o3,floorImpl:i3,gatherNdImpl:a3,gatherV2Impl:l3,greaterEqualImpl:c3,greaterImpl:u3,lessEqualImpl:f3,lessImpl:h3,linSpaceImpl:d3,logImpl:p3,maxImpl:m3,maximumImpl:g3,minimumImpl:y3,multiplyImpl:Tp,negImpl:b3,notEqualImpl:w3,prodImpl:v3,raggedGatherImpl:A3,raggedRangeImpl:C3,raggedTensorToTensorImpl:$3,rangeImpl:N3,rsqrtImpl:D3,scatterImpl:O3,sigmoidImpl:M3,simpleAbsImpl:K_,sliceImpl:P3,sparseFillEmptyRowsImpl:R3,sparseReshapeImpl:L3,sparseSegmentReductionImpl:B3,sqrtImpl:F3,squaredDifferenceImpl:U3,staticRegexReplaceImpl:z3,stridedSliceImpl:V3,stringNGramsImpl:W3,stringSplitImpl:H3,stringToHashBucketFastImpl:j3,subImpl:K3,tileImpl:X3,topKImpl:Y3,transposeImpl:x3,uniqueImpl:Q3});/**
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
 */const{concatImpl:J3,maxImpl:eS,prodImpl:tS,sliceImpl:nS,transposeImpl:sS}=Z3;/**
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
 */class rS{constructor(e,t){this.variableNames=["source"],this.workPerThread=1,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=t,this.rank=t.length,this.dispatchLayout=Kt(this.outputShape),this.dispatch=Ze(this.dispatchLayout,this.outputShape,this.workgroupSize,[this.workPerThread,1,1]),this.start=e,this.uniforms=`start : ${ht(e.length)}, `,this.shaderKey="slice"}getUserCode(){const e=ht(this.rank),t=iS(this.rank);let n;return this.start.length===1?n=this.outputShape.map((i,o)=>"sourceLoc = uniforms.start + coords;"):n=this.outputShape.map((i,o)=>`sourceLoc.${Pa[o]} = uniforms.start.${is(o)} + coords.${Pa[o]};`),`
      ${je("index")} {
        if (index < uniforms.size) {
          var sourceLoc : ${e};
          let coords = getCoordsFromIndex(index);
          ${n.join(`
`)}
          setOutputAtIndex(index, getSource(${t}));
        }
      }
    `}}const Pa=["x","y","z","w","u","v"];function iS(s){if(s===1)return"sourceLoc";if(s<=6)return Pa.slice(0,s).map(e=>`sourceLoc.${e}`).join(",");throw Error(`Slicing for rank ${s} is not yet supported`)}/**
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
 */function oS(s){const{inputs:e,backend:t,attrs:n}=s,{x:r}=e,{begin:i,size:o}=n,[a,l]=rx(r,i,o);if(tx(r,a,l),t.shouldExecuteOnCPU([r])||r.dtype==="string"){const h=t.tensorMap.get(r.dataId),d=nS(h.values,a,l,r.shape,r.dtype);return t.makeTensorInfo(l,r.dtype,d)}if(he(l)===0)return t.makeTensorInfo(l,r.dtype,[]);const u=new rS(a,l),c=[{type:"int32",data:a}];return t.runWebGPUProgram(u,[r],r.dtype,c)}const aS={kernelName:xh,backendName:"webgpu",kernelFunc:oS};/**
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
 */var Ae;(function(s){s[s.ADD=0]="ADD",s[s.ATAN2=1]="ATAN2",s[s.COMPLEX_MULTIPLY_IMAG=2]="COMPLEX_MULTIPLY_IMAG",s[s.COMPLEX_MULTIPLY_REAL=3]="COMPLEX_MULTIPLY_REAL",s[s.DIV=4]="DIV",s[s.ELU_DER=5]="ELU_DER",s[s.EQUAL=6]="EQUAL",s[s.FLOOR_DIV=7]="FLOOR_DIV",s[s.GREATER=8]="GREATER",s[s.GREATER_EQUAL=9]="GREATER_EQUAL",s[s.LESS=10]="LESS",s[s.LESS_EQUAL=11]="LESS_EQUAL",s[s.LOGICAL_AND=12]="LOGICAL_AND",s[s.LOGICAL_OR=13]="LOGICAL_OR",s[s.MAX=14]="MAX",s[s.MIN=15]="MIN",s[s.MOD=16]="MOD",s[s.MUL=17]="MUL",s[s.NOT_EQUAL=18]="NOT_EQUAL",s[s.POW=19]="POW",s[s.PRELU=20]="PRELU",s[s.SQUARED_DIFFERENCE=21]="SQUARED_DIFFERENCE",s[s.SUB=22]="SUB"})(Ae||(Ae={}));const lS="let resultTemp = a + b;",uS="let resultTemp = atan2(a, b);",cS="let resultTemp = areal * breal - aimag * bimag;",hS="let resultTemp = areal * bimag + aimag * breal;",fS="let resultTemp = a / b;",dS="let resultTemp = select(a * (b + 1.0), a, b >= b - b);",pS=`
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a == b);
`,mS=`
  let remainder =
      select(a % b, round(a % b), (round(a) == a) & (round(b) == b));
  let quotient = (a - remainder) / b;
  let resultTemp =
      round(select(quotient, quotient - 1, sign(remainder) == -sign(b)));
`,gS=`
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a > b);
`,yS=`
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a >= b);
`,bS=`
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a < b);
`,wS=`
  let zero = sign(a) * 0 + 0;
  let one = sign(b) * 0 + 1;
  let resultTemp = select(zero, one, a <= b);
`,xS="return f32(a >= 1.0 && b >= 1.0);",vS=`return (vec4<f32>(a >= vec4<f32>(1.0)) *
  vec4<f32>(b >= vec4<f32>(1.0)));`,_S="return f32(a >= 1.0 || b >= 1.0);",SS=`return min(vec4<f32>(a >= vec4<f32>(1.0)) +
  vec4<f32>(b >= vec4<f32>(1.0)), vec4<f32>(1.0));`,IS="let resultTemp = max(a, b);",kS="let resultTemp = min(a, b);",TS=`
  let isNaN = b == 0.;
  var resultTemp = a % b;
  resultTemp = select((resultTemp + b) % b, resultTemp,
      (a < 0. && b < 0.) || (a >= 0. && b > 0.));
`,ES=`
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
`,AS="let resultTemp = a * b;",CS=`
  var resultTemp = f32(a != b);
  let valueForNaN = 1.0;
`,$S=`
  var resultTemp = vec4<f32>(a != b);
  let valueForNaN = 1.0;
`,NS=`
  let isNaN = a < 0.0 && floor(b) < b;
  if (b == 0.0) {
    return 1.0;
  }
  var resultTemp = select(sign(a) * pow(abs(a), b), pow(abs(a), b),
      round(abs(b) % 2.0) != 1.0);
`,DS=`
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
`,OS="if (a < 0.0) { return b * a; }  return a;",MS=`
  let aLessThanZero = vec4<f32>(a < vec4<f32>(0.0));
  return (aLessThanZero * (b * a)) + ((vec4<f32>(1.0) - aLessThanZero) * a);
`,PS="let resultTemp = (a - b) * (a - b);",RS="let resultTemp = a - b;";function LS(s,e){let t;do{switch(s){case Ae.ATAN2:t=uS;break;case Ae.MAX:t=IS;break;case Ae.MIN:t=kS;break;case Ae.MOD:t=e?ES:TS;break;case Ae.NOT_EQUAL:t=e?$S:CS;break;case Ae.POW:t=e?DS:NS;break;default:continue}let n,r,i;return e?(n="isnanVec4",r="vec4<f32>",i="vec4<bool>"):(n="isnan",r="f32",i="bool"),`
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
    `}while(!1);switch(s){case Ae.ADD:t=lS;break;case Ae.COMPLEX_MULTIPLY_IMAG:t=hS;break;case Ae.COMPLEX_MULTIPLY_REAL:t=cS;break;case Ae.DIV:t=fS;break;case Ae.ELU_DER:t=dS;break;case Ae.EQUAL:t=pS;break;case Ae.FLOOR_DIV:t=mS;break;case Ae.GREATER:t=gS;break;case Ae.GREATER_EQUAL:t=yS;break;case Ae.LESS:t=bS;break;case Ae.LESS_EQUAL:t=wS;break;case Ae.LOGICAL_AND:return e?vS:xS;case Ae.LOGICAL_OR:return e?SS:_S;case Ae.MUL:t=AS;break;case Ae.PRELU:return e?MS:OS;case Ae.SQUARED_DIFFERENCE:t=PS;break;case Ae.SUB:t=RS;break}return`
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
 */var ce;(function(s){s[s.ABS=0]="ABS",s[s.ACOS=1]="ACOS",s[s.ACOSH=2]="ACOSH",s[s.ASIN=3]="ASIN",s[s.ASINH=4]="ASINH",s[s.ATAN=5]="ATAN",s[s.ATANH=6]="ATANH",s[s.CEIL=7]="CEIL",s[s.COS=8]="COS",s[s.COSH=9]="COSH",s[s.ELU=10]="ELU",s[s.ERF=11]="ERF",s[s.EXP=12]="EXP",s[s.EXPM1=13]="EXPM1",s[s.FLOOR=14]="FLOOR",s[s.IS_FINITE=15]="IS_FINITE",s[s.IS_INF=16]="IS_INF",s[s.IS_NAN=17]="IS_NAN",s[s.LINEAR=18]="LINEAR",s[s.LOG=19]="LOG",s[s.LOG1P=20]="LOG1P",s[s.LOGICAL_NOT=21]="LOGICAL_NOT",s[s.NEG=22]="NEG",s[s.RELU=23]="RELU",s[s.RELU6=24]="RELU6",s[s.LEAKYRELU=25]="LEAKYRELU",s[s.RECIPROCAL=26]="RECIPROCAL",s[s.ROUND=27]="ROUND",s[s.RSQRT=28]="RSQRT",s[s.SELU=29]="SELU",s[s.SIGMOID=30]="SIGMOID",s[s.SIGN=31]="SIGN",s[s.SIN=32]="SIN",s[s.SINH=33]="SINH",s[s.SOFTPLUS=34]="SOFTPLUS",s[s.SQRT=35]="SQRT",s[s.SQUARE=36]="SQUARE",s[s.STEP=37]="STEP",s[s.TAN=38]="TAN",s[s.TANH=39]="TANH",s[s.TO_INT=40]="TO_INT"})(ce||(ce={}));const BS="return abs(a);",FS=`
  if (abs(a) > 1.) {
    return uniforms.NAN;
  }
  return acos(a);
`,US=`
  if (a < 1.) {
    return uniforms.NAN;
  }
  return acosh(a);
`,zS=`
  if (abs(a) > 1.) {
    return uniforms.NAN;
  }
  return asin(a);
`,VS="return asinh(a);",GS=`
  if (isnan(a)) {
    return uniforms.NAN;
  }
  return atan(a);
`,WS=`
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
`,qS="return ceil(a);",HS="return cos(a);",jS=`
  let e2x = exp(-a);
  return (e2x + 1.0 / e2x) / 2.0;
`,KS="return exp(a) - 1.0;",XS="if (a >= 0.0) { return a; }  return (exp(a) - 1.0);",YS=`
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
`,QS=`
  // Error function is calculated approximately with elementary function.
  // See "Handbook of Mathematical Functions with Formulas,
  // Graphs, and Mathematical Tables", Abramowitz and Stegun.
  let p = ${mx};
  let a1 = ${gx};
  let a2 = ${yx};
  let a3 = ${bx};
  let a4 = ${wx};
  let a5 = ${xx};

  let sign = sign(a);
  let absA = abs(a);
  let t = 1.0 / (1.0 + p * absA);
  return sign * (1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-absA * absA));
`,ZS="return exp(a);",JS="return floor(a);",eI="return f32(!isnan(a) && !isinf(a));",tI="return f32(isinf(a));",nI="return f32(isnan(a));",sI="return a;",rI=`if (a < 0.0) { return uniforms.NAN; }
  return log(a);`,iI=`
  if (isnan(a)) { return a; }
  return log(1.0 + a);
`,oI="return f32(!(a >= 1.0));",aI="return -a;",lI="if (a < 0.0) { return uniforms.alpha * a; } return a;",uI=`
  let aLessThanZero = vec4<f32>(a < vec4<f32>(0.0));
  return (aLessThanZero * (uniforms.alpha * a)) + ((vec4<f32>(1.0) - aLessThanZero) * a);
`,cI="return 1.0 / a;",hI="return select(a, 0.0, a < 0.0);",fI="return clamp(a, 0.0, 6.0);",dI="return clamp(a, vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(6.0, 6.0, 6.0, 6.0));",pI=`
  return select(a, vec4<f32>(0.0), a < vec4<f32>(0.0));
`,mI="return round(a);",gI="return inverseSqrt(a);",yI=`
  if (a >= 0.0) {
    return ${px} * a;
  } else {
    return ${dx} * (exp(a) - 1.0);
  }
`,bI="return 1.0 / (1.0 + exp(-1.0 * a));",wI="return sign(a);",xI="return sin(a);",vI=`
  let e2x = exp(a);
  return (e2x - 1.0 / e2x) / 2.0;
`,_I=`
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
`,SI="return sqrt(a);",II="return a * a;",kI=`
  if (isnan(a)) {
    return a;
  }

  return select(uniforms.stepAlpha, 1.0, a > 0.0);
`,TI="return tan(a);",EI=`
  let e2x = exp(-2.0 * abs(a));
  return sign(a) * (1.0 - e2x) / (1.0 + e2x);
`,AI="return f32(i32((a)));";function ws(s,e){switch(s){case ce.ABS:return BS;case ce.ACOS:return FS;case ce.ACOSH:return US;case ce.ASIN:return zS;case ce.ASINH:return VS;case ce.ATAN:return GS;case ce.ATANH:return WS;case ce.COS:return HS;case ce.COSH:return jS;case ce.CEIL:return qS;case ce.ELU:return e?YS:XS;case ce.ERF:return QS;case ce.EXP:return ZS;case ce.EXPM1:return KS;case ce.FLOOR:return JS;case ce.IS_FINITE:return eI;case ce.IS_INF:return tI;case ce.IS_NAN:return nI;case ce.LINEAR:return sI;case ce.LOG:return rI;case ce.LOG1P:return iI;case ce.LOGICAL_NOT:return oI;case ce.NEG:return aI;case ce.LEAKYRELU:return e?uI:lI;case ce.RECIPROCAL:return cI;case ce.RELU:return e?pI:hI;case ce.RELU6:return e?dI:fI;case ce.ROUND:return mI;case ce.RSQRT:return gI;case ce.SELU:return yI;case ce.SIGMOID:return bI;case ce.SIGN:return wI;case ce.SIN:return xI;case ce.SINH:return vI;case ce.SOFTPLUS:return _I;case ce.SQRT:return SI;case ce.SQUARE:return II;case ce.STEP:return kI;case ce.TAN:return TI;case ce.TANH:return EI;case ce.TO_INT:return AI;default:throw new Error(`BinaryType ${s} is not implemented!`)}}/**
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
 */function Ys(s,e=!1,t=!1,n=3){if(s===null)return"";let r="";if(s==="linear")r=ws(ce.LINEAR);else if(s==="relu")r=ws(ce.RELU,t);else if(s==="elu")r=ws(ce.ELU,t);else if(s==="relu6")r=ws(ce.RELU6,t);else if(s==="prelu")r=LS(Ae.PRELU,t);else if(s==="sigmoid")r=ws(ce.SIGMOID,t);else if(s==="leakyrelu")r=ws(ce.LEAKYRELU,t);else throw new Error(`Activation ${s} has not been implemented for the WebGPU backend.`);const o=me(t?4:1);let a="";return e?a=`
      fn activation(a : ${o}, coords : vec${n}<i32>) -> ${o} {
        let b = getPreluActivationWeightsByOutputCoords(coords);
        ${r}
      }`:a=`
      fn activation(a : ${o}, coords : vec${n}<i32>) -> ${o} {
        ${r}
      }`,a}function Ao(s,e){return`
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
 */function Ap(s,e,t=!1,n=!1,r=!1,i=1){P(s&&i===1||!s,()=>`transposeA ${s} is not compatible with component size ${i}`);const o=`
      ${s?"value = getA(batch, col, row);":"value = getA(batch, row, col);"}

    `,a=e?"value = getB(batch, col, row);":"value = getB(batch, row, col);";return`
  fn mm_readA(batch: i32, row: i32, col: i32) -> ${me(i)} {
    var value = ${me(i)}(0.0);
    ${t&&r?o:`
    ${s?"if(row < uniforms.dimAOuter && col < uniforms.dimInner)":"if(row < uniforms.aShape[1] && col < uniforms.aShape[2])"}
    {
      ${o}
    }
    `}
    return value;
  }

  fn mm_readB(batch: i32, row: i32, col: i32) -> ${me(i)} {
    var value = ${me(i)}(0.0);
    ${a}
    return value;
  }
  `}function Bl(s,e,t,n,r=!1,i=!1,o=!1,a=1){return`
  ${Ap(t,n,r,i,o,a)}
  fn mm_write(batch: i32, row: i32, col: i32, valueIn: ${me(a)}) {
    ${r&&i?"":"if (row < uniforms.dimAOuter && col < uniforms.dimBOuter)"}
    {
      var value = valueIn;
      let coords = vec3<i32>(batch, row, col);
      ${Ao(s,e)}
      setOutputAtCoords(coords[0], coords[1], coords[2], value);
    }
  }
  `}const CI=(s,e)=>s?`
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          kStart + inputRow,
          globalRowStart + inputCol * ${e});
        `:`
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          globalRow + innerRow,
          kStart + inputCol * ${e});
        `,$I=(s,e,t,n)=>{if(s)return`
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
      }`}};function Fl(s,e,t=!1,n=32,r=!1,i=32,o=!1){const a=e[1]*s[1],l=e[0]*s[0],u=t?a:n,c=t?n:a,h=u/e[0],d=n/e[1],w=s[1],I=s[0];return P((t&&h===4&&s[1]===4||!t&&(h===3||h===4))&&u%e[0]===0&&n%e[1]===0&&s[0]===4,()=>`If transposeA ${t} is true, innerElementSize ${h} and workPerThread[1] ${s[1]} must be 4.
          Otherwise, innerElementSize ${h} must be 3 or 4.
      tileAWidth ${u} must be divisible by workgroupSize[0]${e[0]}. tileInner ${n} must be divisible by workgroupSize[1] ${e[1]}. colPerThread ${s[0]} must be 4.`),`
  var<workgroup> mm_Asub : array<array<vec${h}<f32>, ${u/h}>, ${c}>;
  var<workgroup> mm_Bsub : array<array<vec4<f32>, ${l/s[0]}>, ${n}>;

  ${je()} {
    let localRow = i32(localId.y);
    let tileRow = localRow * ${w};
    let tileCol = i32(localId.x);

    let globalRow = i32(globalId.y) * ${w};
    let globalCol = i32(globalId.x) * ${I};
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
            ${CI(t,h)}
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
        ${$I(t,h,w,n)}
        workgroupBarrier();
    }

    for (var innerRow = 0; innerRow < ${w}; innerRow++) {
        mm_write(batch, globalRow + innerRow, globalCol, acc[innerRow]);
    }
  }`}const Nc=s=>s?`
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          kStart + inputRow,
          globalRowStart + inputCol);
        `:`
        mm_Asub[inputRow][inputCol] = mm_readA(batchA,
          globalRowStart + inputRow,
          kStart + inputCol);
        `,NI=s=>s?"let ACached = mm_Asub[k][tileRow + innerRow];":"let ACached = mm_Asub[tileRow + innerRow][k];";function Ul(s,e,t=!1,n=32,r=!1,i=32,o=!1,a=!1){const l=s[1]*e[1],u=s[0]*e[0],c=t?l:n,h=t?n:l;P(h%e[1]===0&&c%e[0]===0&&n%e[1]===0,()=>`tileAHight ${h} must be divisible by workgroupSize[1]${e[1]}, tileAWidth ${c} must be divisible by workgroupSize[0]${e[0]}, tileInner ${n} must be divisible by workgroupSize[1]${e[1]}`);const d=h/e[1],w=c/e[0],I=n/e[1],E=s[1],m=s[0],S=o?`
      let localRow = i32(localId.y);
      let localCol = i32(localId.x);
      let globalRowStart = i32(workgroupId.y) * ${l};
      let globalColStart = i32(workgroupId.x) * ${u};

      // Loop over shared dimension.
      for (var t = 0; t < numTiles; t++) {
        // Load one tile of A into local memory.
        for (var inputRow = localRow; inputRow < ${h}; inputRow = inputRow + ${e[1]}) {
          for (var inputCol = localCol; inputCol < ${c}; inputCol = inputCol + ${e[0]}) {
            ${Nc(t)}
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
          for (var innerRow = 0; innerRow < ${E}; innerRow++) {
            let ACached = ${t?`mm_Asub[k][localRow + innerRow * ${e[1]}];`:`mm_Asub[localRow + innerRow * ${e[1]}][k];`}
            for (var innerCol = 0; innerCol < ${m}; innerCol++) {
              acc[innerRow][innerCol] =
                  fma(ACached, BCached[innerCol], acc[innerRow][innerCol]);
            }
          }
        }
        workgroupBarrier();
      }
      for (var innerRow = 0; innerRow < ${E}; innerRow++) {
        let gRow = globalRowStart + localRow + innerRow * ${e[1]};
        for (var innerCol = 0; innerCol < ${m}; innerCol++) {
          let gCol = globalColStart + localCol + innerCol * ${e[0]};
          mm_write(batch, gRow, gCol, acc[innerRow][innerCol]);
        }
      }
      `:`
  let tileRow = i32(localId.y) * ${E};
  let tileCol = i32(localId.x) * ${m};

  let globalRow = i32(globalId.y) * ${E};
  let globalCol = i32(globalId.x) * ${m};
  let globalRowStart = i32(workgroupId.y) * ${l};

  let tileRowA = i32(localId.y) * ${d};
  let tileColA = i32(localId.x) * ${w};
  let tileRowB = i32(localId.y) * ${I};
  // Loop over shared dimension.
  for (var t = 0; t < numTiles; t++) {
    // Load one tile of A into local memory.
    for (var innerRow = 0; innerRow < ${d}; innerRow++) {
      for (var innerCol = 0; innerCol < ${w}; innerCol++) {
        let inputRow = tileRowA + innerRow;
        let inputCol = tileColA + innerCol;
        ${Nc(t)}
      }
    }

    // Load one tile of B into local memory.
    for (var innerRow = 0; innerRow < ${I}; innerRow++) {
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

      for (var innerRow = 0; innerRow < ${E}; innerRow++) {
        ${NI(t)}
        for (var innerCol = 0; innerCol < ${m}; innerCol++) {
          acc[innerRow][innerCol] =
              fma(ACached, BCached[innerCol], acc[innerRow][innerCol]);
        }
      }
    }

    workgroupBarrier();
  }

  for (var innerRow = 0; innerRow < ${E}; innerRow++) {
    for (var innerCol = 0; innerCol < ${m}; innerCol++) {
      mm_write(batch, globalRow + innerRow, globalCol + innerCol,
          acc[innerRow][innerCol]);
    }
  }
  `;return`
    var<workgroup> mm_Asub : array<array<f32, ${c}>, ${h}>;
    var<workgroup> mm_Bsub : array<array<f32, ${u}>, ${n}>;

    ${je()} {
      let batch = ${r?"0":"i32(globalId.z)"};
      let batchA = ${r||!a?"batch":"batch % uniforms.aShape[0]"};
      let batchB = ${r||!a?"batch":"batch % uniforms.bShape[0]"};
      let numTiles = ${r?`${Math.ceil(i/n)}`:`(uniforms.dimInner - 1) / ${n} + 1`};
      var kStart = ${r?`i32(globalId.z) * ${i}`:"0"};

      var acc : array<array<f32, ${m}>, ${E}>;

      // Without this initialization strange values show up in acc.
      for (var innerRow = 0; innerRow < ${E}; innerRow++) {
        for (var innerCol = 0; innerCol < ${m}; innerCol++) {
          acc[innerRow][innerCol] = 0.0;
        }
      }
      ${S}
    }
  `}const DI=s=>s?`
      mm_readA(batchA, colA, globalRow),
      mm_readA(batchA, colA + 1, globalRow),
      mm_readA(batchA, colA + 2, globalRow),
      mm_readA(batchA, colA + 3, globalRow)
  `:`
      mm_readA(batchA, globalRow, colA),
      mm_readA(batchA, globalRow, colA + 1),
      mm_readA(batchA, globalRow, colA + 2),
      mm_readA(batchA, globalRow, colA + 3)
  `;function OI(s,e=!1){P(s[1]===1&&s[2]===1,()=>`A linear work group size is required. But got ${s}.`);const t=s[0]*4;return`
    var<workgroup> mm_Asub : array<vec4<f32>, ${s[0]}>;

    ${je()} {
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
        mm_Asub[tileCol] = vec4<f32>(${DI(e)});
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
  `}class MI{constructor(e,t,n=!1,r=!1,i=null,o=null,a=null,l=!1){this.variableNames=["A","B"],this.uniforms="dimAOuter : i32, dimBOuter : i32, dimInner : i32,",this.outputShape=t,this.dispatchLayout={x:[2],y:[1],z:[0]};const u=n?e[1]:e[2];if(this.isVec4=(u%4===0&&!n||t[1]%4===0&&n)&&t[2]%4===0&&!r,this.outputComponent=this.isVec4?4:1,this.isVectorA=t[1]===1&&!n,!this.isVec4&&this.isVectorA)this.elementsPerThread=[1,1,1],this.workgroupSize=[32,1,1];else{const d=P_(t[1],u,t[2],n);this.workgroupSize=d.workgroupSize,this.elementsPerThread=d.elementsPerThread}this.dispatch=Ze(this.dispatchLayout,this.outputShape,this.workgroupSize,this.elementsPerThread);const c=i!=null,h=a!=null;c&&this.variableNames.push("bias"),h&&this.variableNames.push("preluActivationWeights"),this.sequentialAccessByThreads=l,this.transposeA=n,this.transposeB=r,this.addBias=c,this.activation=o,this.hasPreluActivationWeights=h,[this.fitAOuter,this.fitBOuter,this.fitInner]=this.getShapeFit(t[1],t[2],u),this.shaderKey=`matMulPacked_${this.elementsPerThread}_${n}_${r}_${this.activation}_${this.fitAOuter}_${this.fitBOuter}_${this.fitInner}_${this.isVec4}_${this.isVectorA}_${this.sequentialAccessByThreads}`}getShapeFit(e,t,n){const r=this.workgroupSize[1]*this.elementsPerThread[1],i=this.workgroupSize[0]*this.elementsPerThread[0];!this.isVec4&&this.isVectorA?this.tileInner=this.workgroupSize[0]*4:this.tileInner=i;const o=e%r===0,a=t%i===0,l=n%this.tileInner===0;return[o,a,l]}getUserCode(){return`
      ${Ys(this.activation,this.hasPreluActivationWeights,this.isVec4)}
      ${Bl(this.addBias,this.activation,!1,this.transposeB,this.fitAOuter,this.fitBOuter,this.fitInner,this.isVec4?4:1)}
      ${this.isVec4?Fl(this.elementsPerThread,this.workgroupSize,this.transposeA,this.tileInner,!1,null,!0):this.isVectorA?OI(this.workgroupSize,this.transposeA):Ul(this.elementsPerThread,this.workgroupSize,this.transposeA,this.tileInner,!1,null,this.sequentialAccessByThreads,!0)}
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
 */function PI(s,e,t,n,r=!1,i=null,o=!1,a=4,l=4,u=4){const c=$=>{switch($){case 1:return"resData = f32(x[xIndex]);";case 3:return"resData = vec3<f32>(x[xIndex], x[xIndex + 1], x[xIndex + 2]);";case 4:return"resData = vec4<f32>(x[xIndex / 4]);";default:throw new Error(`innerElementSize ${$} is not supported.`)}},h=$=>{switch($){case 1:return"return f32(W[row * uniforms.wShape[3] + col]);";case 4:return"return vec4<f32>(W[(row * uniforms.wShape[3] + col) / 4]);";default:throw new Error(`innerElementSize ${$} is not supported.`)}},d=s?`
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
      `,I=s?"uniforms.xShape[1]":"uniforms.xShape[2]",E=s?"uniforms.xShape[2]":"uniforms.xShape[3]",m=s?"row":"col",S=s?"col":"row",b=`
      let inChannels = uniforms.wShape[2];
      let outWidth = ${s?"uniforms.outShape[2]":"uniforms.outShape[3]"};
      let outRow = ${m} / outWidth;
      let outCol = ${m} % outWidth;

      let WRow = ${S} / (uniforms.filterDims[1] * inChannels);
      let WCol = ${S} / inChannels % uniforms.filterDims[1];
      let xRow = outRow * uniforms.strides[0] + uniforms.dilations[0] * WRow - uniforms.pads[0];
      let xCol = outCol * uniforms.strides[1] + uniforms.dilations[1] * WCol - uniforms.pads[1];
      let xCh = ${S} % inChannels;
      var resData = ${me(a)}(0.0);
      // The bounds checking is always needed since we use it to pad zero for
      // the 'same' padding type.
      if (xRow >= 0 && xRow < ${I} && xCol >= 0 && xCol < ${E}) {
        ${d}
        let xIndex = getIndexFromCoords4D(coord, uniforms.xShape);
        ${c(a)}
      }
      return resData;`,f=s?e&&n?`
      ${b}`:`
      if (row < uniforms.dimAOuter && col < uniforms.dimInner) {
        ${b}
      }
      return ${me(a)}(0.0);`:n&&t?`
      ${b}`:`
      if (row < uniforms.dimInner && col < uniforms.dimBOuter) {
        ${b}
      }
      return ${me(a)}(0.0);`,_=`${h(l)}`,v=me(u),T=me(s?a:l),N=me(s?l:a);return`
      ${Ys(i,o,u===4,4)}
      fn mm_readA(batch: i32, row : i32, col : i32) -> ${T} {
        ${s?f:_}
      }

      fn mm_readB(batch: i32, row : i32, col : i32) -> ${N} {
        ${s?_:f}
      }

      fn mm_write(batch: i32, row : i32, col : i32, valueIn : ${v}) {
        if (row < uniforms.dimAOuter && col < uniforms.dimBOuter)
        {
        var value = valueIn;
        let outWidth = ${s?"uniforms.outShape[2]":"uniforms.outShape[3]"};
        ${w}
        ${Ao(r,i)}
        setOutputAtCoords(coords[0], coords[1], coords[2], coords[3], value);
        }
      }`}class RI{constructor(e,t,n,r,i=!1,o=null,a=!1,l=!1){this.variableNames=["x","W"],this.uniforms="filterDims : vec2<i32>, pads : vec2<i32>, strides : vec2<i32>, dilations : vec2<i32>, dimAOuter : i32, dimBOuter : i32, dimInner : i32,",this.outputShape=e.outShape,this.isChannelsLast=e.dataFormat==="channelsLast",this.isVec4=((e.inChannels%4===0||e.inChannels%3===0)&&this.isChannelsLast||e.outWidth%4===0&&!this.isChannelsLast)&&e.outChannels%4===0,this.dispatchLayout=this.isChannelsLast?{x:[3],y:[1,2],z:[0]}:{x:[2,3],y:[1],z:[0]},this.workgroupSize=R_(this.dispatchLayout,this.outputShape,this.isVec4),this.elementsPerThread=L_(this.dispatchLayout,this.outputShape,this.isVec4),this.dispatch=Ze(this.dispatchLayout,this.outputShape,this.workgroupSize,this.elementsPerThread),this.isVec4?(this.outputComponent=4,this.isChannelsLast&&e.inChannels%4!==0?(this.innerElementSize=3,this.variableComponents=[1,4]):(this.innerElementSize=4,this.variableComponents=[4,4]),i&&(this.variableNames.push("bias"),this.variableComponents.push(4)),a&&(this.variableNames.push("preluActivationWeights"),this.variableComponents.push(4))):(this.innerElementSize=this.elementsPerThread[0],i&&this.variableNames.push("bias"),a&&this.variableNames.push("preluActivationWeights")),this.sequentialAccessByThreads=l,this.addBias=i,this.activation=o,this.hasPreluActivationWeights=a,this.tileAOuter=this.workgroupSize[1]*this.elementsPerThread[1],this.tileBOuter=this.workgroupSize[0]*this.elementsPerThread[0],this.tileInner=Math.max(this.workgroupSize[0]*this.innerElementSize,this.workgroupSize[1]),this.fitAOuter=t%this.tileAOuter===0,this.fitBOuter=n%this.tileBOuter===0,this.fitInner=r%this.tileInner===0,this.shaderKey=`conv2DMM_${this.elementsPerThread}_${this.activation}}_${this.fitAOuter}_${this.fitBOuter}_${this.fitInner}_${this.isVec4}_${this.innerElementSize}_${this.isChannelsLast}_${this.sequentialAccessByThreads}`}getUserCode(){const e=this.isVec4?Fl(this.elementsPerThread,this.workgroupSize,!this.isChannelsLast,this.tileInner):Ul(this.elementsPerThread,this.workgroupSize,!this.isChannelsLast,this.tileInner,!1,null,this.sequentialAccessByThreads),t=this.isVec4?[this.innerElementSize,4,4]:[1,1,1];return`
    ${PI(this.isChannelsLast,this.fitAOuter,this.fitBOuter,this.fitInner,this.addBias,this.activation,this.hasPreluActivationWeights,t[0],t[1],t[2])}
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
 */class LI{constructor(e,t=!1,n=null,r=!1){this.variableNames=["x","W"],this.uniforms="filterDims: vec2<i32>, pads: vec2<i32>, strides: vec2<i32>, dilations: vec2<i32>,",this.workgroupSize=[4,4,8],this.outputShape=e.outShape,this.isChannelsLast=e.dataFormat==="channelsLast",this.dispatchLayout=this.isChannelsLast?{x:[2],y:[1],z:[0,3]}:{x:[3],y:[2],z:[0,1]},this.dispatch=Ze(this.dispatchLayout,this.outputShape,this.workgroupSize),this.addBias=t,this.activation=n,this.hasPreluActivationWeights=r,t&&this.variableNames.push("bias"),r&&this.variableNames.push("preluActivationWeights"),this.shaderKey=`conv2dnaive_${this.activation}_${this.isChannelsLast}`}getUserCode(){return`
       ${Ys(this.activation,this.hasPreluActivationWeights,!1,4)}
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
           ${Ao(this.addBias,this.activation)}
           setOutputAtCoords(coords.x, coords.y, coords.z, coords.w, value);
         }
       }
       ${je("index")} {
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
 */class BI{constructor(e,t){this.variableNames=["x"],this.uniforms=`pads : vec2<i32>, strides : vec2<i32>, dilations : vec2<i32>, outWidth : i32, itemsPerBlockRow : i32,
       inChannels : i32,`,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=Kt(this.outputShape),this.dispatch=Ze(this.dispatchLayout,this.outputShape,this.workgroupSize),this.isChannelsLast=t,this.shaderKey=`im2col_${this.isChannelsLast}`}getUserCode(){const e=this.isChannelsLast?1:2,t=this.isChannelsLast?2:3,n=this.isChannelsLast?"coords[1]":"coords[2]",r=this.isChannelsLast?"coords[2]":"coords[1]",i=this.isChannelsLast?"getX(batch, xRow, xCol, ch)":"getX(batch, ch, xRow, xCol)";return`
    ${je("index")} {
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
 */function FI(s){return`
    var<workgroup> sumValues : array<f32, ${s}>;
    ${je()} {
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
  `}class UI{constructor(e,t=!1,n=!1,r=null,i=null,o=null){this.variableNames=["A","B"],this.uniforms="dimAOuter : i32, dimBOuter : i32, dimInner : i32,",this.workgroupSize=[256,1,1],this.outputShape=e,this.dispatchLayout={x:[],y:[1,2],z:[0]},this.dispatch=Ze(this.dispatchLayout,this.outputShape,this.workgroupSize);const a=r!=null,l=o!=null;a&&this.variableNames.push("bias"),l&&this.variableNames.push("preluActivationWeights"),this.transposeA=t,this.transposeB=n,this.addBias=a,this.activation=i,this.hasPreluActivationWeights=l,this.shaderKey=`matMulReduce_${this.activation}_${t}_${n}`}getUserCode(){return`
      ${Ys(this.activation,this.hasPreluActivationWeights)}
      ${Bl(this.addBias,this.activation,this.transposeA,this.transposeB)}
      ${FI(this.workgroupSize[0])}
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
 */function zI(s){const e=s[1],t=s[0],n=e>t?e:t;return`
  var<workgroup> mm_Asub : array<array<f32, ${n}>, ${e}>;
  var<workgroup> mm_Bsub : array<array<f32, ${t}>, ${n}>;

  // If the output size is small for matrix multiplication, avoid to use vec4
  // and handle some elements per thread to optimally utilize the ALU.
  // Read data from global memory to registers firstly, then store them into
  // shared memory, so it is instruction-Level parallelism for arithmetic
  // operations and others handle IO operations between barrier api, makes ALU
  // and load/store units work simultaneously, could improves the performance.
  ${je()} {
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
  `}class VI{constructor(e,t,n,r=!1,i=!1,o=null,a=null,l=null){this.variableNames=["A","B"],this.uniforms="dimAOuter : i32, dimBOuter : i32, dimInner : i32,",this.workgroupSize=[16,8,1],this.outputShape=n,this.dispatchLayout={x:[2],y:[1],z:[0]},this.dispatch=[Math.ceil(n[2]/this.workgroupSize[0]),Math.ceil(n[1]/this.workgroupSize[1]),n[0]];const u=o!=null;u&&this.variableNames.push("bias");const c=l!=null;c&&this.variableNames.push("preluActivationWeights"),this.transposeA=r,this.transposeB=i,this.addBias=u,this.activation=a,this.hasPreluActivationWeights=c,this.shaderKey=`matMulSmallOutputSize_${this.activation}_${r}_${i}`}getUserCode(){return`
      ${Ys(this.activation,this.hasPreluActivationWeights)}
      ${Bl(this.addBias,this.activation,this.transposeA,this.transposeB)}
      ${zI(this.workgroupSize)}
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
 */class GI{constructor(e,t,n=!1,r=!1){this.variableNames=["A","B"],this.uniforms="dimAOuter : i32, dimBOuter : i32, dimInner : i32,",this.workgroupSize=[8,8,1],this.atomic=!0,this.splitedDimInner=128,P(e[0]===1,()=>"MatMulSplitKProgram only supports batch = 1."),this.outputShape=e,this.dispatchLayout={x:[2],y:[1],z:[0,3]};const i=(n&&this.outputShape[1]%4===0||!n&&t%4===0)&&this.outputShape[2]%4===0;this.elementsPerThread=[4,4,this.splitedDimInner],this.outputComponent=i?4:1,i||(this.outputShape[1]<16&&(this.elementsPerThread[1]=1),this.outputShape[2]<16&&(this.elementsPerThread[0]=1)),this.dispatch=Ze(this.dispatchLayout,[this.outputShape[0],this.outputShape[1],this.outputShape[2],t],this.workgroupSize,this.elementsPerThread),this.transposeA=n,this.transposeB=r,this.shaderKey=`matMulSplitK_${n}_${r}_${this.elementsPerThread}_${this.outputComponent}`}getUserCode(){const e=this.outputComponent;return`
      ${Ap(!1,this.transposeB,!1,!1,!1,e)}
      fn mm_write(batch: i32, row : i32, col : i32, value : ${me(e)}) {
        if (row < uniforms.dimAOuter && col < uniforms.dimBOuter) {
          let coords = vec3<i32>(batch, row, col);
          let flatIndex = getOutputIndexFromCoords(coords);
          // The problem is that we should initialize output to zero before using.
          // Otherwise, the original value will be added to the result.
          for (var i = 0; i < ${e}; i = i + 1) {
            ${__("&result[flatIndex + i]",`${e>1?"value[i]":"value"}`)}
          }
        }
      }
      ${e===4?Fl(this.elementsPerThread,this.workgroupSize,this.transposeA,32,!0,this.splitedDimInner):Ul(this.elementsPerThread,this.workgroupSize,this.transposeA,32,!0,this.splitedDimInner)}
    `}}class WI{constructor(e,t=null,n=null,r=null){this.uniforms="",this.variableNames=["x"],this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=e,this.dispatchLayout=Kt(this.outputShape),this.dispatch=Ze(this.dispatchLayout,this.outputShape,this.workgroupSize),this.addBias=t!=null,this.hasPreluActivationWeights=r!=null,this.activation=n,this.addBias&&this.variableNames.push("bias"),this.hasPreluActivationWeights&&this.variableNames.push("preluActivationWeights"),this.shaderKey=`biasActivation_${n}`}getUserCode(){return`
    ${Ys(this.activation,this.hasPreluActivationWeights)}
    ${je("index")} {
      if (index < uniforms.size) {
        let coords = getCoordsFromIndex(index);
        var value = getXByOutputIndex(index);
        ${Ao(this.addBias,this.activation)}
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
 */function Re(s){const{inputs:e,attrs:t}=s,{x:n}=e,{shape:r}=t,i=he(n.shape),o=_m(r,i),a=he(o);return P(i===a,()=>`The new shape (${o}) has ${a} elements and the old shape (${n.shape}) has ${i} elements. The new shape and old shape must have the same number of elements.`),s.backend.incRef(n.dataId),{dataId:n.dataId,shape:o,dtype:n.dtype}}/**
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
 */function Cp({a:s,b:e,transposeA:t,transposeB:n,backend:r,bias:i=null,preluActivationWeights:o=null,leakyreluAlpha:a=0,activation:l=null}){const u=s.shape.length,c=e.shape.length,h=t?s.shape[u-2]:s.shape[u-1],d=e.shape[c-2],w=t?s.shape[u-1]:s.shape[u-2],I=e.shape[c-1],E=s.shape.slice(0,-2),m=e.shape.slice(0,-2),S=he(E),b=he(m),_=St(s.shape.slice(0,-2),e.shape.slice(0,-2)).concat([w,I]);P(h===d,()=>`Error in matMul: inner shapes (${h}) and (${d}) of Tensors with shapes ${s.shape} and ${e.shape} and transposeA=${t} and transposeB=${n} must match.`);const v=t?[S,h,w]:[S,w,h],T=[b,d,I],N=Re({inputs:{x:s},backend:r,attrs:{shape:v}}),O=Re({inputs:{x:e},backend:r,attrs:{shape:T}}),$=[N,O],A=Math.max(S,b),g=[N,O],p=[{type:"int32",data:[w]},{type:"int32",data:[I]},{type:"int32",data:[h]}];let y,x;const k=[A,w,I];let C=fe().get("WEBGPU_MATMUL_PROGRAM_TYPE");if(C<0){const z=fe().getNumber("WEBGPU_THRESHOLD_TO_INCREASE_WORKGROUPS_FOR_MATMUL"),j=z>0?z:r.thresholdToIncreaseWorkgroups,G=A*Math.ceil(w/32)*Math.ceil(I/32);G<=j||w<=8&&G<=j*2?A*w*I<=128?C=cn.MatMulReduceProgram:A===1&&d>=2e3?C=cn.MatMulSplitKProgram:C=cn.MatMulSmallOutputSizeProgram:C=cn.MatMulPackedProgram}switch(C){case cn.MatMulReduceProgram:y=new UI(k,t,n,i,l,o);break;case cn.MatMulSplitKProgram:{if(x=kp({backend:r,attrs:{shape:k,value:0,dtype:s.dtype}}),y=new GI(k,d,t,n),i||l){x=r.runWebGPUProgram(y,g,s.dtype,p,x);const j=new WI(x.shape,i,l,o);let G=null;const X=[x];i&&X.push(i),o&&X.push(o),l==="leakyrelu"&&(G=[{type:"float32",data:[a]}],j.uniforms+=" alpha : f32,");const Z=r.runWebGPUProgram(j,X,x.dtype,G);$.push(x);const ne=Re({inputs:{x:Z},backend:r,attrs:{shape:_}});$.push(Z);for(const oe of $)r.disposeData(oe.dataId);return ne}break}case cn.MatMulSmallOutputSizeProgram:y=new VI(v,T,k,t,n,i,l,o);break;case cn.MatMulPackedProgram:const z=r.adapterInfo.isIntel();y=new MI(v,k,t,n,i,l,o,z);break;default:throw new Error(`Unsupported MatMulProgramType ${C}.`)}i&&g.push(i),o&&g.push(o),l==="leakyrelu"&&(p.push({type:"float32",data:[a]}),y.uniforms+=" alpha : f32,"),x=r.runWebGPUProgram(y,g,s.dtype,p,x);const R=Re({inputs:{x},backend:r,attrs:{shape:_}});$.push(x);for(const z of $)r.disposeData(z.dataId);return R}/**
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
 */function so(s,e){const t=s.length;return t>=3?e?[...s.slice(0,-3),s[t-3]*s[t-2],s[t-1]]:[...s.slice(0,-3),s[t-3],s[t-2]*s[t-1]]:!e&&t===1&&s[0]>1?[s[0],1]:null}function qI({x:s,filter:e,convInfo:t,backend:n,bias:r=null,preluActivationWeights:i=null,leakyreluAlpha:o=0,activation:a=null}){const l=t.dataFormat==="channelsLast",u=!l,c=!1,h=l&&t.filterHeight===t.inHeight&&t.filterWidth===t.inWidth&&t.padInfo.type==="VALID",d=[];let w,I;if(h){const S=t.inHeight*t.inWidth*t.inChannels;w=Re({inputs:{x:s},backend:n,attrs:{shape:[1,t.batchSize,S]}}),I=Re({inputs:{x:e},backend:n,attrs:{shape:[1,S,t.outChannels]}})}else w=Re({inputs:{x:s},backend:n,attrs:{shape:l?[t.batchSize,t.inHeight*t.inWidth,t.inChannels]:[t.batchSize,t.inChannels,t.inHeight*t.inWidth]}}),I=Re({inputs:{x:e},backend:n,attrs:{shape:[1,t.inChannels,t.outChannels]}});if(d.push(w),d.push(I),i!=null){const S=so(i.shape,l);S!=null&&(i=Re({inputs:{x:i},backend:n,attrs:{shape:S}}),d.push(i))}if(r!=null){const S=so(r.shape,l);S!=null&&(r=Re({inputs:{x:r},backend:n,attrs:{shape:S}}),d.push(r))}const E=Cp({a:l?w:I,b:l?I:w,transposeA:u,transposeB:c,backend:n,bias:r,activation:a,preluActivationWeights:i,leakyreluAlpha:o}),m=Re({inputs:{x:E},backend:n,attrs:{shape:t.outShape}});d.push(E);for(const S of d)n.disposeData(S.dataId);return m}function HI({x:s,filter:e,convInfo:t,backend:n,bias:r=null,preluActivationWeights:i=null,leakyreluAlpha:o=0,activation:a=null}){const{filterWidth:l,filterHeight:u,inChannels:c,strideWidth:h,strideHeight:d,padInfo:w,outWidth:I,outHeight:E,dilationWidth:m,dilationHeight:S,dataFormat:b}=t,f=b==="channelsLast",_=l*u*c,v=E*I,T=f?[t.batchSize,v,_]:[t.batchSize,_,v],N=new BI(T,f),O=[{type:"int32",data:[w.top,w.left]},{type:"int32",data:[d,h]},{type:"int32",data:[S,m]},{type:"int32",data:[I]},{type:"int32",data:[c*l]},{type:"int32",data:[c]}],$=n.runWebGPUProgram(N,[s],s.dtype,O),A=[];A.push($);const g=Re({inputs:{x:e},backend:n,attrs:{shape:[1,_,-1]}});if(A.push(g),i!=null){const C=so(i.shape,f);C!=null&&(i=Re({inputs:{x:i},backend:n,attrs:{shape:C}}),A.push(i))}if(r!=null){const C=so(r.shape,f);C!=null&&(r=Re({inputs:{x:r},backend:n,attrs:{shape:C}}),A.push(r))}const x=Cp({a:f?$:g,b:f?g:$,transposeA:!f,transposeB:!1,backend:n,bias:r,activation:a,preluActivationWeights:i,leakyreluAlpha:o}),k=Re({inputs:{x},backend:n,attrs:{shape:t.outShape}});A.push(x);for(const C of A)n.disposeData(C.dataId);return k}function jI({x:s,filter:e,convInfo:t,backend:n,bias:r=null,preluActivationWeights:i=null,leakyreluAlpha:o=0,activation:a=null}){const l=r!=null,u=i!=null,c=t.dataFormat==="channelsLast",h=c&&t.filterHeight===t.inHeight&&t.filterWidth===t.inWidth&&t.padInfo.type==="VALID",d=fe().getBool("WEBGPU_USE_NAIVE_CONV2D_DEBUG");if(!d&&(h||t.filterHeight===1&&t.filterWidth===1&&t.dilationHeight===1&&t.dilationWidth===1&&t.strideHeight===1&&t.strideWidth===1&&(t.padInfo.type==="SAME"||t.padInfo.type==="VALID")))return qI({x:s,filter:e,convInfo:t,backend:n,bias:r,activation:a,preluActivationWeights:i,leakyreluAlpha:o});const w=fe().getNumber("WEBGPU_THRESHOLD_TO_INCREASE_WORKGROUPS_FOR_MATMUL"),I=w>-1?w:n.thresholdToIncreaseWorkgroups,E=t.batchSize*Math.ceil(t.outHeight*t.outWidth/32)*Math.ceil(t.outChannels/32);if(fe().getBool("WEBGPU_CONV_SEPARATE_IM2COL_SHADER")||E<=I)return HI({x:s,filter:e,convInfo:t,backend:n,bias:r,preluActivationWeights:i,leakyreluAlpha:o,activation:a});let m;const S=[t.padInfo.top,t.padInfo.left],b=[{type:"int32",data:[t.filterHeight,t.filterWidth]},{type:"int32",data:[...S]},{type:"int32",data:[t.strideHeight,t.strideWidth]},{type:"int32",data:[t.dilationHeight,t.dilationWidth]}];if(d)m=new LI(t,l,a,u);else{const T=c?t.outHeight*t.outWidth:t.outChannels,N=c?t.outChannels:t.outHeight*t.outWidth,O=t.filterHeight*t.filterWidth*t.inChannels;b.push({type:"int32",data:[T]},{type:"int32",data:[N]},{type:"int32",data:[O]});const $=n.adapterInfo.isIntel();m=new RI(t,T,N,O,l,a,u,$)}const f=[],_=[s,e];l&&(!c&&r.shape.length===1&&(r=Re({inputs:{x:r},backend:n,attrs:{shape:[r.shape[0],1,1]}}),f.push(r)),_.push(r)),u&&(!c&&i.shape.length===1&&(i=Re({inputs:{x:i},backend:n,attrs:{shape:[i.shape[0],1,1]}}),f.push(i)),_.push(i)),a==="leakyrelu"&&(b.push({type:"float32",data:[o]}),m.uniforms+=" alpha : f32,");const v=n.runWebGPUProgram(m,_,s.dtype,b);for(const T of f)n.disposeData(T.dataId);return v}/**
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
 */function KI(s){const{inputs:e,backend:t,attrs:n}=s,{x:r,filter:i,bias:o,preluActivationWeights:a}=e,{strides:l,pad:u,dataFormat:c,dilations:h,dimRoundingMode:d,activation:w,leakyreluAlpha:I}=n,E=E0(c),m=ja(r.shape,i.shape,l,h,u,d,!1,E);return jI({x:r,filter:i,convInfo:m,backend:t,bias:o,preluActivationWeights:a,leakyreluAlpha:I,activation:w})}const XI={kernelName:ua,backendName:"webgpu",kernelFunc:KI};/**
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
 */class YI{constructor(e){this.variableNames=["x"],this.uniforms="strides : vec2<i32>,",this.workgroupSize=[256,1,1],this.size=!0,this.outputShape=e.outShape,this.dispatchLayout=Kt(this.outputShape),this.dispatch=Ze(this.dispatchLayout,this.outputShape,this.workgroupSize),this.shaderKey="poolWithFilterSizeEqualsOne"}getUserCode(){return`
      ${je("index")} {
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
 */class QI{constructor(e,t,n=!1,r=!1,i=!1){if(this.variableNames=["x"],this.uniforms="strides : vec2<i32>, pads : vec2<i32>, dilations : vec2<i32>, convDims : vec2<i32>, filterDims : vec2<i32>,",this.workgroupSize=[128,1,1],this.size=!0,t==="avg"&&n)throw new Error("Cannot compute positions for average pool.");this.outputShape=e.outShape,this.dispatchLayout=Kt(this.outputShape),this.dispatch=Ze(this.dispatchLayout,this.outputShape,this.workgroupSize),this.poolType=t,this.computePositions=n,this.flattenPositions=r,this.includeBatchIndex=i,this.shaderKey=`pool2D_${t}_${n}_${r}_${i}`}getUserCode(){let e;this.poolType==="avg"?e="resultValue = resultValue + value; count = count + 1.0;":this.computePositions?e=`let currMaxValue = mix(value, maxValue, maxValueFound);
      if (value >= currMaxValue) {
        maxValue = value;
        maxValueFound = 1.0;
        maxPosition = ${this.flattenPositions?this.includeBatchIndex?"((batch * uniforms.xShape[1] + xR) * uniforms.xShape[2] + xC) * uniforms.xShape[3] + d":"(xR * uniforms.xShape[2] + xC) * uniforms.xShape[3] + d":"wR * uniforms.filterDims.y + wC"};
      }`:e="resultValue = max(value, resultValue);";let t="resultValue";return this.poolType==="avg"&&(t="resultValue / max(count, 1.0)"),`
      ${je("index")} {
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
 */class ZI{constructor(e,t){this.variableNames=["A"],this.workgroupSize=[16,16,1];const n=new Array(e.length);for(let r=0;r<n.length;r++)n[r]=e[t[r]];this.outputShape=n,this.dispatchLayout={x:[0],y:[1]},this.dispatch=Ze(this.dispatchLayout,this.outputShape,this.workgroupSize,[1,1,1]),this.shaderKey="transposeShared"}getUserCode(){P(this.workgroupSize[0]===this.workgroupSize[1],()=>`Must be a square tile, current tile shape is ${this.workgroupSize[0]} x ${this.workgroupSize[1]}`);const e=this.workgroupSize[0];return`
      var<workgroup> tile : array<array<f32, ${this.workgroupSize[0]+1}>, ${this.workgroupSize[0]}>;
      ${je()} {
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
 */class JI{constructor(e,t){this.variableNames=["A"],this.workPerThread=1,this.workgroupSize=[64,1,1],this.size=!0;const n=new Array(e.length);for(let r=0;r<n.length;r++)n[r]=e[t[r]];this.outputShape=n,this.dispatchLayout=Kt(this.outputShape),this.dispatch=Ze(this.dispatchLayout,this.outputShape,this.workgroupSize,[this.workPerThread,1,1]),this.newDim=t,this.shaderKey=`transpose_${t}`}getUserCode(){const e=ht(this.outputShape.length),t=ek(this.newDim);return`
      ${je("index")} {
        for(var i = 0; i < ${this.workPerThread}; i = i + 1) {
          let flatIndex = index * ${this.workPerThread} + i;
          if(flatIndex < uniforms.size) {
            let coords = getCoordsFromIndex(flatIndex);
            setOutputAtIndex(flatIndex, A[getIndexFromCoords${this.outputShape.length}D(
              ${e}(${t}), uniforms.aShape)]);
          }
        }
      }
    `}}function ek(s){const e=s.length;if(e>6)throw Error(`Transpose for rank ${e} is not yet supported`);const t=new Array(e);for(let n=0;n<s.length;n++)t[s[n]]=`coords.${is(n)}`;return t.join()}/**
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
 */function tk(s){const{inputs:e,backend:t,attrs:n}=s,{x:r}=e,{perm:i}=n,o=t,a=r.shape.length,l=new Array(a);for(let c=0;c<l.length;c++)l[c]=r.shape[i[c]];if(t.shouldExecuteOnCPU([r])){const h=o.tensorMap.get(r.dataId).values,d=sS(h,r.shape,r.dtype,i,l);return t.makeTensorInfo(l,r.dtype,d)}if(r.shape.length===2&&Ht(i,[1,0])){const c=new ZI(r.shape,i);return o.runWebGPUProgram(c,[r],r.dtype)}const u=new JI(r.shape,i);return o.runWebGPUProgram(u,[r],r.dtype)}/**
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
 */class nk{constructor(e,t,n){this.variableNames=["x"],this.uniforms="reduceSize : i32,",this.size=!0,this.inputShape=[e.batchSize,e.inSize];const[r]=Qa(this.inputShape,[1]);this.outputShape=r.length===0?[1]:r,e.inSize>=32768&&n>=512?this.workgroupSize=[512,1,1]:e.inSize>=4096?this.workgroupSize=[256,1,1]:this.workgroupSize=[64,1,1],this.dispatchLayout=Kt(this.outputShape),this.dispatch=Ze(this.dispatchLayout,this.outputShape,[1,1,1]),this.reduceType=t,this.shaderKey=`reduce_${t}`}getUserCode(){let e="",t="0.0";const n=this.workgroupSize[0];this.reduceType==="min"||this.reduceType==="max"?(e=`
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
       ${je("index")} {
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
 */const sk={mean:"float32",all:"bool",any:"bool"};function rk(s,e,t,n,r){const i=s.shape.length,o=[],a=Ar(e,s.shape);let l=a;const u=c1(l,i);let c=s;u!=null&&(c=tk({inputs:{x:s},attrs:{perm:u},backend:r}),l=h1(l.length,i),o.push(c)),u1(n,l,i);const[h,d]=Qa(c.shape,l);let w=h;t&&(w=jh(h,a));let I;if(r.shouldExecuteOnCPU([c])){const E=r.tensorMap.get(c.dataId).values;switch(n){case"max":const m=eS(E,he(d),w,s.dtype);I=r.makeTensorInfo(w,s.dtype,m);break;case"prod":const{outVals:S,outShape:b,outDtype:f}=tS(c.shape,c.dtype,E,l);I=r.makeTensorInfo(b,f,S);break;default:throw new Error(`${n} CPU implementation is not yet supported.`)}}else{const E=he(d),S=he(c.shape)/E,b={windowSize:E,inSize:E,batchSize:S,outSize:1},f=sk[n]||Ty(s.dtype),_=[{type:"int32",data:[E]}],v=new nk(b,n,r.device.limits.maxComputeWorkgroupSizeX),T=r.runWebGPUProgram(v,[c],f,_);o.push(T),I=Re({inputs:{x:T},attrs:{shape:w},backend:r})}return o.forEach(E=>r.disposeData(E.dataId)),I}/**
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
 */function ik(s){const{inputs:e,backend:t,attrs:n}=s,{x:r}=e,{reductionIndices:i,keepDims:o}=n;return rk(r,i,o,"max",t)}/**
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
 */function ok(s,e,t,n){if(e.filterWidth===1&&e.filterHeight===1&&Ht(e.inShape,e.outShape))return Wn({inputs:{x:s},backend:n});if(e.filterWidth===e.inWidth&&e.filterHeight===e.inHeight&&e.batchSize===1&&e.padInfo.type==="VALID"){const o=s.shape.length,a=Re({inputs:{x:s},backend:n,attrs:{shape:[s.shape[o-3]*s.shape[o-2],s.shape[o-1]]}});let l;P(t==="max",()=>`Invalid pool type ${t}`),l=ik({inputs:{x:a},backend:n,attrs:{reductionIndices:0,keepDims:!1}});const u=Re({inputs:{x:l},backend:n,attrs:{shape:e.outShape}});return n.disposeData(a.dataId),n.disposeData(l.dataId),u}let r;const i=[{type:"int32",data:[e.strideHeight,e.strideWidth]}];return e.filterHeight===1&&e.filterWidth===1?r=new YI(e):(P(t==="max",()=>`Invalid pool type ${t}`),r=new QI(e,"max"),i.push({type:"int32",data:[e.padInfo.top,e.padInfo.left]},{type:"int32",data:[e.dilationHeight,e.dilationWidth]},{type:"int32",data:[e.inHeight,e.inWidth]},{type:"int32",data:[e.effectiveFilterHeight,e.effectiveFilterWidth]})),n.runWebGPUProgram(r,[s],s.dtype,i)}/**
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
 */function ak(s){const{inputs:e,backend:t,attrs:n}=s,{x:r}=e,{filterSize:i,strides:o,pad:a,dimRoundingMode:l}=n,c=S0(r.shape,i,o,1,a,l);return ok(r,c,"max",t)}const lk={kernelName:yh,backendName:"webgpu",kernelFunc:ak};/**
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
 */class uk{constructor(e,t,n,r){this.variableNames=["x"],this.uniforms="adjustHeightWidth : vec2<f32>, roundBase : f32,",this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=[e[0],t,n,e[3]],this.dispatchLayout=Kt(this.outputShape),this.dispatch=Ze(this.dispatchLayout,this.outputShape,this.workgroupSize),this.halfPixelCenters=r,this.shaderKey=`resizeNearest_${r}`}getUserCode(){let e;return this.halfPixelCenters?e="max((vec2<f32>(rc) + vec2<f32>(0.5)) * effectiveInputOverOutputRatioRC, vec2<f32>(0.0))":e="vec2<f32>(rc) * effectiveInputOverOutputRatioRC",`
      ${je("index")} {
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
 */function ck(s){const{inputs:e,backend:t,attrs:n}=s,{images:r}=e,{alignCorners:i,halfPixelCenters:o,size:a}=n,[l,u]=a,c=i&&l>1?1:0,h=i&&u>1?1:0,w=[{type:"float32",data:[c,h]},{type:"float32",data:[i?.5:0]}],I=new uk(r.shape,l,u,o);return t.runWebGPUProgram(I,[r],r.dtype,w)}const hk={kernelName:wh,backendName:"webgpu",kernelFunc:ck};/**
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
 */class fk{constructor(e){this.uniforms="",this.workPerThread=1,this.workgroupSize=[64,1,1],this.size=!0,this.outputShape=wr(e,1),this.variableNames=e.map((t,n)=>`T${n}`),this.dispatchLayout=Kt(this.outputShape),this.dispatch=Ze(this.dispatchLayout,this.outputShape,this.workgroupSize,[this.workPerThread,1,1]),this.offsetLength=e.length-1;for(let t=0;t<this.offsetLength;t++)this.uniforms+=`offset${t} : i32,`;this.shaderKey="concat"}getUserCode(){const e=[];if(this.offsetLength>0){e.push("if (yC < uniforms.offset0){ setOutputAtCoords(coords.x, coords.y, getT0(yR, yC)); }");for(let i=1;i<this.offsetLength;i++)e.push(`else if (yC < uniforms.offset${[i]}){ setOutputAtCoords(coords.x, coords.y, getT${i}(yR, yC - uniforms.offset${i-1})); }`);const n=this.offsetLength,r=this.offsetLength-1;e.push(`else { setOutputAtCoords(coords.x, coords.y, getT${n}(yR, yC - uniforms.offset${r})); }`)}else e.push("setOutputAtCoords(coords.x, coords.y, getT0(yR, yC));");return`
      ${je("index")} {
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
 */function dk(s){const{inputs:e,backend:t}=s,{real:n,imag:r}=e,i=t.makeTensorInfo(n.shape,"complex64"),o=t.tensorMap.get(i.dataId),a=Wn({inputs:{x:n},backend:t}),l=Wn({inputs:{x:r},backend:t});return o.complexTensorInfos={real:a,imag:l},i}/**
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
 */function pk(s){const{inputs:e,backend:t}=s,{input:n}=e,r=t.tensorMap.get(n.dataId);return Wn({inputs:{x:r.complexTensorInfos.imag},backend:t})}/**
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
 */function mk(s){const{inputs:e,backend:t}=s,{input:n}=e,r=t.tensorMap.get(n.dataId);return Wn({inputs:{x:r.complexTensorInfos.real},backend:t})}/**
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
 */function ar(s,e,t){const n=s[0].dtype;if(n==="complex64"){const I=s.map(f=>mk({inputs:{input:f},backend:t})),E=s.map(f=>pk({inputs:{input:f},backend:t})),m=ar(I,e,t),S=ar(E,e,t),b=dk({inputs:{real:m,imag:S},backend:t});return I.forEach(f=>t.disposeData(f.dataId)),E.forEach(f=>t.disposeData(f.dataId)),t.disposeData(m.dataId),t.disposeData(S.dataId),b}let r=t.shouldExecuteOnCPU(s);if(n==="string"&&(r=!0),r){const I=s.map(v=>{const N=[-1,he(v.shape.slice(e))];return Re({inputs:{x:v},backend:t,attrs:{shape:N}})}),E=I.map(v=>({vals:t.readSync(v.dataId),shape:v.shape})),m=wr(I.map(v=>v.shape),1),S=I[0].shape[0]===1,b=J3(E,m,n,S),f=wr(s.map(v=>v.shape),e),_=t.makeTensorInfo(f,n,b);return I.forEach(v=>t.disposeData(v.dataId)),_}const i=t.device.limits.maxStorageBuffersPerShaderStage-1;if(s.length>i){const I=[];for(let m=0;m<s.length;m+=i){const S=s.slice(m,m+i);I.push(ar(S,e,t))}const E=ar(I,e,t);for(const m of I)t.disposeData(m.dataId);return E}const{tensors2D:o,outShape:a}=gk(s,e,t),l=o.map(I=>I.shape),u=new fk(l),c=[],h=new Array(l.length-1);if(h.length>0){h[0]=l[0][1],c.push({type:"int32",data:[h[0]]});for(let I=1;I<h.length;I++)h[I]=h[I-1]+l[I][1],c.push({type:"int32",data:[h[I]]})}const d=t.runWebGPUProgram(u,o,o[0].dtype,c);o.forEach(I=>t.disposeData(I.dataId));const w=Re({inputs:{x:d},backend:t,attrs:{shape:a}});return t.disposeData(d.dataId),w}function gk(s,e,t){const n=wr(s.map(i=>i.shape),e);return{tensors2D:s.map(i=>Re({inputs:{x:i},backend:t,attrs:{shape:[he(i.shape.slice(0,e)),he(i.shape.slice(e))]}})),outShape:n}}/**
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
 */function yk(s){const{inputs:e,backend:t,attrs:n}=s,{axis:r}=n,i=Ar(r,e[0].shape)[0],o=e.map(u=>u.shape);lx(o,i);const a=wr(e.map(u=>u.shape),i);if(he(a)===0)return t.makeTensorInfo(a,e[0].dtype,[]);const l=e.filter(u=>he(u.shape)>0);return l.length===1?Wn({inputs:{x:l[0]},backend:t}):ar(l,i,t)}const bk={kernelName:gh,backendName:"webgpu",kernelFunc:yk},wk=[z_,j_,aS,XI,lk,hk,bk,V_];for(const s of wk)ry({...s,backendName:"webgpu-oidn"});async function xk(){try{const s={powerPreference:"high-performance"},e=await navigator.gpu.requestAdapter(s),t={},n=[];e.features.has("timestamp-query")&&n.push("timestamp-query"),e.features.has("bgra8unorm-storage")&&n.push(["bgra8unorm-storage"]),t.requiredFeatures=n;const r=e.limits;t.requiredLimits={maxComputeWorkgroupStorageSize:r.maxComputeWorkgroupStorageSize,maxComputeWorkgroupsPerDimension:r.maxComputeWorkgroupsPerDimension,maxStorageBufferBindingSize:r.maxStorageBufferBindingSize,maxBufferSize:r.maxBufferSize,maxComputeWorkgroupSizeX:r.maxComputeWorkgroupSizeX,maxComputeInvocationsPerWorkgroup:r.maxComputeInvocationsPerWorkgroup};const i=await e.requestDevice(t),o=e.info??await e.requestAdapterInfo?.();return vk(i,o)}catch{}}async function vk(s,e){let t=H.findBackend("webgpu-oidn");return t!=null||(t=new Ur(s,e),H.registerBackend("webgpu-oidn",()=>t),await H.setBackend("webgpu-oidn")),t}async function _k(s,e,t){const n=await xk(),r=mm(s);return new y_(r,n,t)}async function Sk(s,e,t){return fetch(s).then(n=>n.arrayBuffer()).then(n=>_k(n,e,t))}var Ik=`const EPS = 1e-4;\r
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
`,kk=`struct VertexOutput\r
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
`,Tk=""+new URL("../rt_ldr.tza",import.meta.url).href;class Co{#n=0;#t=0;#e=0;#s=1;constructor(e=0,t,n,r=255){typeof t=="number"&&typeof n=="number"?this.RGBA=[e,t,n,r]:(this.#n=(e>>16&255)/255,this.#t=(e>>8&255)/255,this.#e=(255&e)/255,this.#s=r/255)}Set(e,t=255){return this.#n=(e>>16&255)/255,this.#t=(e>>8&255)/255,this.#e=(255&e)/255,this.#s=t/255,this}Premultiply(e,t){t??=new Co,e??=this.#s;const n=this.#n*e,r=this.#t*e,i=this.#e*e;return t.rgba=[n,r,i,e],t}set rgb(e){this.#n=e[0],this.#t=e[1],this.#e=e[2],this.#s=e[3]??1}get rgb(){return[this.#n,this.#t,this.#e]}set a(e){this.#s=e}get a(){return this.#s}set rgba(e){this.rgb=e}get rgba(){return this.rgb.concat(this.#s)}set RGB(e){this.#n=e[0]/255,this.#t=e[1]/255,this.#e=e[2]/255,this.#s=(e[3]??255)/255}get RGB(){return[255*this.#n,255*this.#t,255*this.#e]}set A(e){this.#s=e/255}get A(){return 255*this.#s}set RGBA(e){this.RGB=e}get RGBA(){return this.RGB.concat(this.A)}}Pe({RAD:Math.PI/180,DEG:180/Math.PI,HPI:Math.PI/2,TAU:2*Math.PI});Pe({DEVICE_LOST:"Device::Lost"});const te=Pe({FORMAT_NOT_SUPPORTED:"FORMAT_NOT_SUPPORTED",WEBGPU_NOT_SUPPORTED:"WEBGPU_NOT_SUPPORTED",ADAPTER_NOT_FOUND:"ADAPTER_NOT_FOUND",FEATURE_NOT_FOUND:"FEATURE_NOT_FOUND",DEVICE_NOT_FOUND:"DEVICE_NOT_FOUND",DEVICE_NOT_REQUESTED:"DEVICE_NOT_REQUESTED",DEVICE_LOST:"DEVICE_LOST",SHADER_CODE_NOT_FOUND:"SHADER_CODE_NOT_FOUND",SHADER_MODULE_NOT_FOUND:"SHADER_MODULE_NOT_FOUND",VERTEX_ENTRY_NOT_FOUND:"VERTEX_ENTRY_NOT_FOUND",VERTEX_ATTRIBUTE_NOT_FOUND:"VERTEX_ATTRIBUTE_NOT_FOUND",UNIFORM_NOT_FOUND:"UNIFORM_NOT_FOUND",STORAGE_NOT_FOUND:"STORAGE_NOT_FOUND",INVALID_UNIFORM_NAME:"INVALID_UNIFORM_NAME",BINDING_NOT_FOUND:"BINDING_NOT_FOUND",PIPELINE_NOT_FOUND:"PIPELINE_NOT_FOUND",LEGACY_RENDER_PIPELINE_NOT_FOUND:"LEGACY_RENDER_PIPELINE_NOT_FOUND",RENDERER_NOT_FOUND:"RENDERER_NOT_FOUND",TEXTURE_SIZE_NOT_FOUND:"TEXTURE_SIZE_NOT_FOUND",TEXTURE_NOT_FOUND:"TEXTURE_NOT_FOUND",INVALID_BYTES_PER_ROW:"INVALID_BYTES_PER_ROW",CANVAS_NOT_FOUND:"CANVAS_NOT_FOUND",CONTEXT_NOT_FOUND:"CONTEXT_NOT_FOUND",RENDER_PASS_NOT_FOUND:"RENDER_PASS_NOT_FOUND",COMMAND_ENCODER_NOT_FOUND:"COMMAND_ENCODER_NOT_FOUND",FONT_TEXTURE_NOT_FOUND:"FONT_TEXTURE_NOT_FOUND",TIMESTAMP_QUERY_NOT_FOUND:"TIMESTAMP_QUERY_NOT_FOUND",RENDER_PASS_ENDED:"RENDER_PASS_ENDED"}),$p=Pe({FORMAT_NOT_SUPPORTED:"Format is not yet supported: ",WEBGPU_NOT_SUPPORTED:"WebGPU is not supported in this browser.",ADAPTER_NOT_FOUND:"Failed to get a GPUAdapter.",DEVICE_NOT_FOUND:"Failed to get a GPUDevice.",FEATURE_NOT_FOUND:"Failed to get a GPUFeature ",DEVICE_NOT_REQUESTED:"GPUDevice was not requested.",DEVICE_LOST:"WebGPU device was lost. ",SHADER_CODE_NOT_FOUND:`Failed to get a WGSL shader when creating shader module.
        An empty shader will be used instead.`,SHADER_MODULE_NOT_FOUND:"Failed to get shader module in ",VERTEX_ENTRY_NOT_FOUND:"Failed to find function ",VERTEX_ATTRIBUTE_NOT_FOUND:"Failed to find vertex attribute ",UNIFORM_NOT_FOUND:"Failed to find uniform ",STORAGE_NOT_FOUND:"Failed to find storage ",INVALID_UNIFORM_NAME:"Requested uniform is already in use and managed internally: ",BINDING_NOT_FOUND:"Failed to find binding ",PIPELINE_NOT_FOUND:"Failed to get GPU",LEGACY_RENDER_PIPELINE_NOT_FOUND:'"Device.RenderPipeline" instance is required in `LegacyTexture` for this operation.\n        Pass it to the `LegacyTexture` constructor or use `Texture.LegacyRenderer` setter before ',RENDERER_NOT_FOUND:'"Device.Renderer" instance is required in `Texture` for this operation.\n        Pass it to the `Texture` constructor or use `Texture.Renderer` setter before ',TEXTURE_SIZE_NOT_FOUND:"`size` array or a `width` value is required in `options` parameter of ",TEXTURE_NOT_FOUND:"`options` is required to have a `texture` value or its `create` entry\n        to be either `true` or a `TextureDescriptor` object when calling ",INVALID_BYTES_PER_ROW:"`bytesPerRow` parameter is not a multiple of 256 in ",CANVAS_NOT_FOUND:"Failed to get a WebGPU canvas.",CONTEXT_NOT_FOUND:"Failed to get a WebGPU context.",RENDER_PASS_NOT_FOUND:"Failed to use pipeline in render pass because it has not started.",COMMAND_ENCODER_NOT_FOUND:"Failed to get a GPUCommandEncoder.",FONT_TEXTURE_NOT_FOUND:"Failed to find font texture in ",TIMESTAMP_QUERY_NOT_FOUND:'"timestamp-query" feature is required to be set with\n        `Device.SetRequiredFeatures` when creating a new `GPUTiming` instance.',RENDER_PASS_ENDED:"Failed get a render pass because it has ended.\n        `Render` method has to be called with `submit` flag set to `false`."}),Ek=Pe({WEBGPU_NOT_SUPPORTED:0,ADAPTER_NOT_FOUND:1,DEVICE_NOT_FOUND:2,DEVICE_NOT_REQUESTED:3,DEVICE_LOST:4,CANVAS_NOT_FOUND:5,CONTEXT_NOT_FOUND:6,COMMAND_ENCODER_NOT_FOUND:7,PIPELINE_NOT_FOUND:8});function mt(s,e){console.warn(`${$p[s]}${e??""}`.replace(/\s\s+/g," "))}function ie(s,e){throw new Error(`${$p[s]}${e??""}`.replace(/\s\s+/g," "),{cause:Ek[s]})}class Gt{constructor(e,t){this.name=e,this.attributes=t,this.size=0}get isArray(){return!1}get isStruct(){return!1}get isTemplate(){return!1}get isPointer(){return!1}getTypeName(){return this.name}}class Dc{constructor(e,t,n){this.name=e,this.type=t,this.attributes=n,this.offset=0,this.size=0}get isArray(){return this.type.isArray}get isStruct(){return this.type.isStruct}get isTemplate(){return this.type.isTemplate}get align(){return this.type.isStruct?this.type.align:0}get members(){return this.type.isStruct?this.type.members:null}get format(){return this.type.isArray||this.type.isTemplate?this.type.format:null}get count(){return this.type.isArray?this.type.count:0}get stride(){return this.type.isArray?this.type.stride:this.size}}class Ln extends Gt{constructor(e,t){super(e,t),this.members=[],this.align=0,this.startLine=-1,this.endLine=-1,this.inUse=!1}get isStruct(){return!0}}class Vn extends Gt{constructor(e,t){super(e,t),this.count=0,this.stride=0}get isArray(){return!0}getTypeName(){return`array<${this.format.getTypeName()}, ${this.count}>`}}class Ra extends Gt{constructor(e,t,n){super(e,n),this.format=t}get isPointer(){return!0}getTypeName(){return`&${this.format.getTypeName()}`}}class ps extends Gt{constructor(e,t,n,r){super(e,n),this.format=t,this.access=r}get isTemplate(){return!0}getTypeName(){let e=this.name;if(this.format!==null){if(e==="vec2"||e==="vec3"||e==="vec4"||e==="mat2x2"||e==="mat2x3"||e==="mat2x4"||e==="mat3x2"||e==="mat3x3"||e==="mat3x4"||e==="mat4x2"||e==="mat4x3"||e==="mat4x4"){if(this.format.name==="f32")return e+="f",e;if(this.format.name==="i32")return e+="i",e;if(this.format.name==="u32")return e+="u",e;if(this.format.name==="bool")return e+="b",e;if(this.format.name==="f16")return e+="h",e}e+=`<${this.format.name}>`}else if(e==="vec2"||e==="vec3"||e==="vec4")return e;return e}}var Mn;(s=>{s[s.Uniform=0]="Uniform",s[s.Storage=1]="Storage",s[s.Texture=2]="Texture",s[s.Sampler=3]="Sampler",s[s.StorageTexture=4]="StorageTexture"})(Mn||(Mn={}));class li{constructor(e,t,n,r,i,o,a){this.name=e,this.type=t,this.group=n,this.binding=r,this.attributes=i,this.resourceType=o,this.access=a}get isArray(){return this.type.isArray}get isStruct(){return this.type.isStruct}get isTemplate(){return this.type.isTemplate}get size(){return this.type.size}get align(){return this.type.isStruct?this.type.align:0}get members(){return this.type.isStruct?this.type.members:null}get format(){return this.type.isArray||this.type.isTemplate?this.type.format:null}get count(){return this.type.isArray?this.type.count:0}get stride(){return this.type.isArray?this.type.stride:this.size}}class Ak{constructor(e,t){this.name=e,this.type=t}}class Ck{constructor(e,t,n,r){this.name=e,this.type=t,this.locationType=n,this.location=r,this.interpolation=null}}class Oc{constructor(e,t,n,r){this.name=e,this.type=t,this.locationType=n,this.location=r}}class $k{constructor(e,t,n,r){this.name=e,this.type=t,this.attributes=n,this.id=r}}class Nk{constructor(e,t,n){this.name=e,this.type=t,this.attributes=n}}class Dk{constructor(e,t=null,n){this.stage=null,this.inputs=[],this.outputs=[],this.arguments=[],this.returnType=null,this.resources=[],this.overrides=[],this.startLine=-1,this.endLine=-1,this.inUse=!1,this.calls=new Set,this.name=e,this.stage=t,this.attributes=n}}class Ok{constructor(){this.vertex=[],this.fragment=[],this.compute=[]}}function Mk(s){var e=(32768&s)>>15,t=(31744&s)>>10,n=1023&s;return t==0?(e?-1:1)*Math.pow(2,-14)*(n/Math.pow(2,10)):t==31?n?NaN:1/0*(e?-1:1):(e?-1:1)*Math.pow(2,t-15)*(1+n/Math.pow(2,10))}const Np=new Float32Array(1),Pk=new Int32Array(Np.buffer),ot=new Uint16Array(1);function Rk(s){Np[0]=s;const e=Pk[0],t=e>>31&1;let n=e>>23&255,r=8388607&e;if(n===255)return ot[0]=t<<15|31744|(r!==0?512:0),ot[0];if(n===0){if(r===0)return ot[0]=t<<15,ot[0];r|=8388608;let i=113;for(;!(8388608&r);)r<<=1,i--;return n=127-i,r&=8388607,n>0?(r=(r>>126-n)+(r>>127-n&1),ot[0]=t<<15|n<<10|r>>13,ot[0]):(ot[0]=t<<15,ot[0])}return n=n-127+15,n>=31?(ot[0]=t<<15|31744,ot[0]):n<=0?n<-10?(ot[0]=t<<15,ot[0]):(r=(8388608|r)>>1-n,ot[0]=t<<15|r>>13,ot[0]):(r>>=13,ot[0]=t<<15|n<<10|r,ot[0])}const zl=new Uint32Array(1),Dp=new Float32Array(zl.buffer,0,1);function Mc(s){const e=112+(s>>6&31)<<23|(63&s)<<17;return zl[0]=e,Dp[0]}function ye(s,e,t,n){const r=[0,0,0,0];for(let i=0;i<n;++i)switch(t){case"8unorm":r[i]=s[e]/255,e++;break;case"8snorm":r[i]=s[e]/255*2-1,e++;break;case"8uint":r[i]=s[e],e++;break;case"8sint":r[i]=s[e]-127,e++;break;case"16uint":r[i]=s[e]|s[e+1]<<8,e+=2;break;case"16sint":r[i]=(s[e]|s[e+1]<<8)-32768,e+=2;break;case"16float":r[i]=Mk(s[e]|s[e+1]<<8),e+=2;break;case"32uint":case"32sint":r[i]=s[e]|s[e+1]<<8|s[e+2]<<16|s[e+3]<<24,e+=4;break;case"32float":r[i]=new Float32Array(s.buffer,e,1)[0],e+=4}return r}function ve(s,e,t,n,r){for(let i=0;i<n;++i)switch(t){case"8unorm":s[e]=255*r[i],e++;break;case"8snorm":s[e]=127.5*(r[i]+1),e++;break;case"8uint":s[e]=r[i],e++;break;case"8sint":s[e]=r[i]+127,e++;break;case"16uint":new Uint16Array(s.buffer,e,1)[0]=r[i],e+=2;break;case"16sint":new Int16Array(s.buffer,e,1)[0]=r[i],e+=2;break;case"16float":{const o=Rk(r[i]);new Uint16Array(s.buffer,e,1)[0]=o,e+=2;break}case"32uint":new Uint32Array(s.buffer,e,1)[0]=r[i],e+=4;break;case"32sint":new Int32Array(s.buffer,e,1)[0]=r[i],e+=4;break;case"32float":new Float32Array(s.buffer,e,1)[0]=r[i],e+=4}return r}const ta={r8unorm:{bytesPerBlock:1,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},r8snorm:{bytesPerBlock:1,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},r8uint:{bytesPerBlock:1,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},r8sint:{bytesPerBlock:1,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},rg8unorm:{bytesPerBlock:2,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rg8snorm:{bytesPerBlock:2,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rg8uint:{bytesPerBlock:2,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rg8sint:{bytesPerBlock:2,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rgba8unorm:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},"rgba8unorm-srgb":{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rgba8snorm:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rgba8uint:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rgba8sint:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},bgra8unorm:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},"bgra8unorm-srgb":{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},r16uint:{bytesPerBlock:2,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},r16sint:{bytesPerBlock:2,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},r16float:{bytesPerBlock:2,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},rg16uint:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rg16sint:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rg16float:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rgba16uint:{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rgba16sint:{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rgba16float:{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},r32uint:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},r32sint:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},r32float:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:1},rg32uint:{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rg32sint:{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rg32float:{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!1,channels:2},rgba32uint:{bytesPerBlock:16,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rgba32sint:{bytesPerBlock:16,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rgba32float:{bytesPerBlock:16,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rgb10a2uint:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rgb10a2unorm:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},rg11b10ufloat:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},stencil8:{bytesPerBlock:1,blockWidth:1,blockHeight:1,isCompressed:!1,isDepthStencil:!0,hasDepth:!1,hasStencil:!0,channels:1},depth16unorm:{bytesPerBlock:2,blockWidth:1,blockHeight:1,isCompressed:!1,isDepthStencil:!0,hasDepth:!0,hasStencil:!1,channels:1},depth24plus:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,isDepthStencil:!0,hasDepth:!0,hasStencil:!1,depthOnlyFormat:"depth32float",channels:1},"depth24plus-stencil8":{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!1,isDepthStencil:!0,hasDepth:!0,hasStencil:!0,depthOnlyFormat:"depth32float",channels:1},depth32float:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,isDepthStencil:!0,hasDepth:!0,hasStencil:!1,channels:1},"depth32float-stencil8":{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!1,isDepthStencil:!0,hasDepth:!0,hasStencil:!0,stencilOnlyFormat:"depth32float",channels:1},rgb9e5ufloat:{bytesPerBlock:4,blockWidth:1,blockHeight:1,isCompressed:!1,channels:4},"bc1-rgba-unorm":{bytesPerBlock:8,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"bc1-rgba-unorm-srgb":{bytesPerBlock:8,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"bc2-rgba-unorm":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"bc2-rgba-unorm-srgb":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"bc3-rgba-unorm":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"bc3-rgba-unorm-srgb":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"bc4-r-unorm":{bytesPerBlock:8,blockWidth:4,blockHeight:4,isCompressed:!0,channels:1},"bc4-r-snorm":{bytesPerBlock:8,blockWidth:4,blockHeight:4,isCompressed:!0,channels:1},"bc5-rg-unorm":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:2},"bc5-rg-snorm":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:2},"bc6h-rgb-ufloat":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"bc6h-rgb-float":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"bc7-rgba-unorm":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"bc7-rgba-unorm-srgb":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"etc2-rgb8unorm":{bytesPerBlock:8,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"etc2-rgb8unorm-srgb":{bytesPerBlock:8,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"etc2-rgb8a1unorm":{bytesPerBlock:8,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"etc2-rgb8a1unorm-srgb":{bytesPerBlock:8,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"etc2-rgba8unorm":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"etc2-rgba8unorm-srgb":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"eac-r11unorm":{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!0,channels:1},"eac-r11snorm":{bytesPerBlock:8,blockWidth:1,blockHeight:1,isCompressed:!0,channels:1},"eac-rg11unorm":{bytesPerBlock:16,blockWidth:1,blockHeight:1,isCompressed:!0,channels:2},"eac-rg11snorm":{bytesPerBlock:16,blockWidth:1,blockHeight:1,isCompressed:!0,channels:2},"astc-4x4-unorm":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"astc-4x4-unorm-srgb":{bytesPerBlock:16,blockWidth:4,blockHeight:4,isCompressed:!0,channels:4},"astc-5x4-unorm":{bytesPerBlock:16,blockWidth:5,blockHeight:4,isCompressed:!0,channels:4},"astc-5x4-unorm-srgb":{bytesPerBlock:16,blockWidth:5,blockHeight:4,isCompressed:!0,channels:4},"astc-5x5-unorm":{bytesPerBlock:16,blockWidth:5,blockHeight:5,isCompressed:!0,channels:4},"astc-5x5-unorm-srgb":{bytesPerBlock:16,blockWidth:5,blockHeight:5,isCompressed:!0,channels:4},"astc-6x5-unorm":{bytesPerBlock:16,blockWidth:6,blockHeight:5,isCompressed:!0,channels:4},"astc-6x5-unorm-srgb":{bytesPerBlock:16,blockWidth:6,blockHeight:5,isCompressed:!0,channels:4},"astc-6x6-unorm":{bytesPerBlock:16,blockWidth:6,blockHeight:6,isCompressed:!0,channels:4},"astc-6x6-unorm-srgb":{bytesPerBlock:16,blockWidth:6,blockHeight:6,isCompressed:!0,channels:4},"astc-8x5-unorm":{bytesPerBlock:16,blockWidth:8,blockHeight:5,isCompressed:!0,channels:4},"astc-8x5-unorm-srgb":{bytesPerBlock:16,blockWidth:8,blockHeight:5,isCompressed:!0,channels:4},"astc-8x6-unorm":{bytesPerBlock:16,blockWidth:8,blockHeight:6,isCompressed:!0,channels:4},"astc-8x6-unorm-srgb":{bytesPerBlock:16,blockWidth:8,blockHeight:6,isCompressed:!0,channels:4},"astc-8x8-unorm":{bytesPerBlock:16,blockWidth:8,blockHeight:8,isCompressed:!0,channels:4},"astc-8x8-unorm-srgb":{bytesPerBlock:16,blockWidth:8,blockHeight:8,isCompressed:!0,channels:4},"astc-10x5-unorm":{bytesPerBlock:16,blockWidth:10,blockHeight:5,isCompressed:!0,channels:4},"astc-10x5-unorm-srgb":{bytesPerBlock:16,blockWidth:10,blockHeight:5,isCompressed:!0,channels:4},"astc-10x6-unorm":{bytesPerBlock:16,blockWidth:10,blockHeight:6,isCompressed:!0,channels:4},"astc-10x6-unorm-srgb":{bytesPerBlock:16,blockWidth:10,blockHeight:6,isCompressed:!0,channels:4},"astc-10x8-unorm":{bytesPerBlock:16,blockWidth:10,blockHeight:8,isCompressed:!0,channels:4},"astc-10x8-unorm-srgb":{bytesPerBlock:16,blockWidth:10,blockHeight:8,isCompressed:!0,channels:4},"astc-10x10-unorm":{bytesPerBlock:16,blockWidth:10,blockHeight:10,isCompressed:!0,channels:4},"astc-10x10-unorm-srgb":{bytesPerBlock:16,blockWidth:10,blockHeight:10,isCompressed:!0,channels:4},"astc-12x10-unorm":{bytesPerBlock:16,blockWidth:12,blockHeight:10,isCompressed:!0,channels:4},"astc-12x10-unorm-srgb":{bytesPerBlock:16,blockWidth:12,blockHeight:10,isCompressed:!0,channels:4},"astc-12x12-unorm":{bytesPerBlock:16,blockWidth:12,blockHeight:12,isCompressed:!0,channels:4},"astc-12x12-unorm-srgb":{bytesPerBlock:16,blockWidth:12,blockHeight:12,isCompressed:!0,channels:4}};class Xt{constructor(){this.id=Xt._id++,this.line=0}get isAstNode(){return!0}get astNodeType(){return""}search(e){e(this)}searchBlock(e,t){if(e){t(ro.instance);for(const n of e)n instanceof Array?this.searchBlock(n,t):n.search(t);t(io.instance)}}constEvaluate(e,t){throw new Error("Cannot evaluate node")}constEvaluateString(e){return this.constEvaluate(e).toString()}}Xt._id=0;class ro extends Xt{}ro.instance=new ro;class io extends Xt{}io.instance=new io;const Op=new Set(["all","all","any","select","arrayLength","abs","acos","acosh","asin","asinh","atan","atanh","atan2","ceil","clamp","cos","cosh","countLeadingZeros","countOneBits","countTrailingZeros","cross","degrees","determinant","distance","dot","dot4U8Packed","dot4I8Packed","exp","exp2","extractBits","faceForward","firstLeadingBit","firstTrailingBit","floor","fma","fract","frexp","insertBits","inverseSqrt","ldexp","length","log","log2","max","min","mix","modf","normalize","pow","quantizeToF16","radians","reflect","refract","reverseBits","round","saturate","sign","sin","sinh","smoothStep","sqrt","step","tan","tanh","transpose","trunc","dpdx","dpdxCoarse","dpdxFine","dpdy","dpdyCoarse","dpdyFine","fwidth","fwidthCoarse","fwidthFine","textureDimensions","textureGather","textureGatherCompare","textureLoad","textureNumLayers","textureNumLevels","textureNumSamples","textureSample","textureSampleBias","textureSampleCompare","textureSampleCompareLevel","textureSampleGrad","textureSampleLevel","textureSampleBaseClampToEdge","textureStore","atomicLoad","atomicStore","atomicAdd","atomicSub","atomicMax","atomicMin","atomicAnd","atomicOr","atomicXor","atomicExchange","atomicCompareExchangeWeak","pack4x8snorm","pack4x8unorm","pack4xI8","pack4xU8","pack4x8Clamp","pack4xU8Clamp","pack2x16snorm","pack2x16unorm","pack2x16float","unpack4x8snorm","unpack4x8unorm","unpack4xI8","unpack4xU8","unpack2x16snorm","unpack2x16unorm","unpack2x16float","storageBarrier","textureBarrier","workgroupBarrier","workgroupUniformLoad","subgroupAdd","subgroupExclusiveAdd","subgroupInclusiveAdd","subgroupAll","subgroupAnd","subgroupAny","subgroupBallot","subgroupBroadcast","subgroupBroadcastFirst","subgroupElect","subgroupMax","subgroupMin","subgroupMul","subgroupExclusiveMul","subgroupInclusiveMul","subgroupOr","subgroupShuffle","subgroupShuffleDown","subgroupShuffleUp","subgroupShuffleXor","subgroupXor","quadBroadcast","quadSwapDiagonal","quadSwapX","quadSwapY"]);class Oe extends Xt{constructor(){super()}}class kr extends Oe{constructor(e,t,n,r,i,o){super(),this.calls=new Set,this.name=e,this.args=t,this.returnType=n,this.body=r,this.startLine=i,this.endLine=o}get astNodeType(){return"function"}search(e){if(this.attributes)for(const t of this.attributes)e(t);e(this);for(const t of this.args)e(t);this.searchBlock(this.body,e)}}class Lk extends Oe{constructor(e){super(),this.expression=e}get astNodeType(){return"staticAssert"}search(e){this.expression.search(e)}}class Mp extends Oe{constructor(e,t){super(),this.condition=e,this.body=t}get astNodeType(){return"while"}search(e){this.condition.search(e),this.searchBlock(this.body,e)}}class La extends Oe{constructor(e,t){super(),this.body=e,this.loopId=t}get astNodeType(){return"continuing"}search(e){this.searchBlock(this.body,e)}}class Pp extends Oe{constructor(e,t,n,r){super(),this.init=e,this.condition=t,this.increment=n,this.body=r}get astNodeType(){return"for"}search(e){var t,n,r;(t=this.init)===null||t===void 0||t.search(e),(n=this.condition)===null||n===void 0||n.search(e),(r=this.increment)===null||r===void 0||r.search(e),this.searchBlock(this.body,e)}}class Cn extends Oe{constructor(e,t,n,r,i){super(),this.attributes=null,this.name=e,this.type=t,this.storage=n,this.access=r,this.value=i}get astNodeType(){return"var"}search(e){var t;e(this),(t=this.value)===null||t===void 0||t.search(e)}}class Vl extends Oe{constructor(e,t,n){super(),this.attributes=null,this.name=e,this.type=t,this.value=n}get astNodeType(){return"override"}search(e){var t;(t=this.value)===null||t===void 0||t.search(e)}}class mr extends Oe{constructor(e,t,n,r,i){super(),this.attributes=null,this.name=e,this.type=t,this.storage=n,this.access=r,this.value=i}get astNodeType(){return"let"}search(e){var t;e(this),(t=this.value)===null||t===void 0||t.search(e)}}class Si extends Oe{constructor(e,t,n,r,i){super(),this.attributes=null,this.name=e,this.type=t,this.storage=n,this.access=r,this.value=i}get astNodeType(){return"const"}constEvaluate(e,t){return this.value.constEvaluate(e,t)}search(e){var t;e(this),(t=this.value)===null||t===void 0||t.search(e)}}var Es,lr,V,B;(s=>{s.increment="++",s.decrement="--"})(Es||(Es={})),(s=>{s.parse=e=>{const t=e;if(t=="parse")throw new Error("Invalid value for IncrementOperator");return s[t]}})(Es||(Es={}));class Rp extends Oe{constructor(e,t){super(),this.operator=e,this.variable=t}get astNodeType(){return"increment"}search(e){this.variable.search(e)}}(s=>{s.assign="=",s.addAssign="+=",s.subtractAssin="-=",s.multiplyAssign="*=",s.divideAssign="/=",s.moduloAssign="%=",s.andAssign="&=",s.orAssign="|=",s.xorAssign="^=",s.shiftLeftAssign="<<=",s.shiftRightAssign=">>="})(lr||(lr={})),(lr||(lr={})).parse=s=>{const e=s;if(e=="parse")throw new Error("Invalid value for AssignOperator");return e};class Lp extends Oe{constructor(e,t,n){super(),this.operator=e,this.variable=t,this.value=n}get astNodeType(){return"assign"}search(e){this.variable.search(e),this.value.search(e)}}class Gl extends Oe{constructor(e,t){super(),this.name=e,this.args=t}get astNodeType(){return"call"}isBuiltin(){return Op.has(this.name)}search(e){for(const t of this.args)t.search(e);e(this)}}class Bp extends Oe{constructor(e,t){super(),this.body=e,this.continuing=t}get astNodeType(){return"loop"}search(e){var t;this.searchBlock(this.body,e),(t=this.continuing)===null||t===void 0||t.search(e)}}class Fp extends Oe{constructor(e,t){super(),this.condition=e,this.cases=t}get astNodeType(){return"switch"}search(e){e(this);for(const t of this.cases)t.search(e)}}class Up extends Oe{constructor(e,t,n,r){super(),this.condition=e,this.body=t,this.elseif=n,this.else=r}get astNodeType(){return"if"}search(e){this.condition.search(e),this.searchBlock(this.body,e),this.searchBlock(this.elseif,e),this.searchBlock(this.else,e)}}class zp extends Oe{constructor(e){super(),this.value=e}get astNodeType(){return"return"}search(e){var t;(t=this.value)===null||t===void 0||t.search(e)}}class Bk extends Oe{constructor(e){super(),this.name=e}get astNodeType(){return"enable"}}class Fk extends Oe{constructor(e){super(),this.extensions=e}get astNodeType(){return"requires"}}class Vp extends Oe{constructor(e,t){super(),this.severity=e,this.rule=t}get astNodeType(){return"diagnostic"}}class Wl extends Oe{constructor(e,t){super(),this.name=e,this.type=t}get astNodeType(){return"alias"}}class Uk extends Oe{constructor(){super()}get astNodeType(){return"discard"}}class Gp extends Oe{constructor(){super(),this.condition=null,this.loopId=-1}get astNodeType(){return"break"}}class Wp extends Oe{constructor(){super(),this.loopId=-1}get astNodeType(){return"continue"}}class K extends Oe{constructor(e){super(),this.attributes=null,this.name=e}get astNodeType(){return"type"}get isStruct(){return!1}get isArray(){return!1}static maxFormatType(e){let t=e[0];if(t.name==="f32")return t;for(let n=1;n<e.length;++n){const r=K._priority.get(t.name);K._priority.get(e[n].name)<r&&(t=e[n])}return t.name==="x32"?K.i32:t}getTypeName(){return this.name}}K.x32=new K("x32"),K.f32=new K("f32"),K.i32=new K("i32"),K.u32=new K("u32"),K.f16=new K("f16"),K.bool=new K("bool"),K.void=new K("void"),K._priority=new Map([["f32",0],["f16",1],["u32",2],["i32",3],["x32",3]]);class Pc extends K{constructor(e){super(e)}}class kn extends K{constructor(e,t,n,r){super(e),this.members=t,this.startLine=n,this.endLine=r}get astNodeType(){return"struct"}get isStruct(){return!0}getMemberIndex(e){for(let t=0;t<this.members.length;t++)if(this.members[t].name==e)return t;return-1}search(e){for(const t of this.members)e(t)}}class U extends K{constructor(e,t,n){super(e),this.format=t,this.access=n}get astNodeType(){return"template"}getTypeName(){let e=this.name;if(this.format!==null){if(e==="vec2"||e==="vec3"||e==="vec4"||e==="mat2x2"||e==="mat2x3"||e==="mat2x4"||e==="mat3x2"||e==="mat3x3"||e==="mat3x4"||e==="mat4x2"||e==="mat4x3"||e==="mat4x4"){if(this.format.name==="f32")return e+="f",e;if(this.format.name==="i32")return e+="i",e;if(this.format.name==="u32")return e+="u",e;if(this.format.name==="bool")return e+="b",e;if(this.format.name==="f16")return e+="h",e}e+=`<${this.format.name}>`}else if(e==="vec2"||e==="vec3"||e==="vec4")return e;return e}}U.vec2f=new U("vec2",K.f32,null),U.vec3f=new U("vec3",K.f32,null),U.vec4f=new U("vec4",K.f32,null),U.vec2i=new U("vec2",K.i32,null),U.vec3i=new U("vec3",K.i32,null),U.vec4i=new U("vec4",K.i32,null),U.vec2u=new U("vec2",K.u32,null),U.vec3u=new U("vec3",K.u32,null),U.vec4u=new U("vec4",K.u32,null),U.vec2h=new U("vec2",K.f16,null),U.vec3h=new U("vec3",K.f16,null),U.vec4h=new U("vec4",K.f16,null),U.vec2b=new U("vec2",K.bool,null),U.vec3b=new U("vec3",K.bool,null),U.vec4b=new U("vec4",K.bool,null),U.mat2x2f=new U("mat2x2",K.f32,null),U.mat2x3f=new U("mat2x3",K.f32,null),U.mat2x4f=new U("mat2x4",K.f32,null),U.mat3x2f=new U("mat3x2",K.f32,null),U.mat3x3f=new U("mat3x3",K.f32,null),U.mat3x4f=new U("mat3x4",K.f32,null),U.mat4x2f=new U("mat4x2",K.f32,null),U.mat4x3f=new U("mat4x3",K.f32,null),U.mat4x4f=new U("mat4x4",K.f32,null),U.mat2x2h=new U("mat2x2",K.f16,null),U.mat2x3h=new U("mat2x3",K.f16,null),U.mat2x4h=new U("mat2x4",K.f16,null),U.mat3x2h=new U("mat3x2",K.f16,null),U.mat3x3h=new U("mat3x3",K.f16,null),U.mat3x4h=new U("mat3x4",K.f16,null),U.mat4x2h=new U("mat4x2",K.f16,null),U.mat4x3h=new U("mat4x3",K.f16,null),U.mat4x4h=new U("mat4x4",K.f16,null),U.mat2x2i=new U("mat2x2",K.i32,null),U.mat2x3i=new U("mat2x3",K.i32,null),U.mat2x4i=new U("mat2x4",K.i32,null),U.mat3x2i=new U("mat3x2",K.i32,null),U.mat3x3i=new U("mat3x3",K.i32,null),U.mat3x4i=new U("mat3x4",K.i32,null),U.mat4x2i=new U("mat4x2",K.i32,null),U.mat4x3i=new U("mat4x3",K.i32,null),U.mat4x4i=new U("mat4x4",K.i32,null),U.mat2x2u=new U("mat2x2",K.u32,null),U.mat2x3u=new U("mat2x3",K.u32,null),U.mat2x4u=new U("mat2x4",K.u32,null),U.mat3x2u=new U("mat3x2",K.u32,null),U.mat3x3u=new U("mat3x3",K.u32,null),U.mat3x4u=new U("mat3x4",K.u32,null),U.mat4x2u=new U("mat4x2",K.u32,null),U.mat4x3u=new U("mat4x3",K.u32,null),U.mat4x4u=new U("mat4x4",K.u32,null);class Ii extends K{constructor(e,t,n,r){super(e),this.storage=t,this.type=n,this.access=r}get astNodeType(){return"pointer"}}class gr extends K{constructor(e,t,n,r){super(e),this.attributes=t,this.format=n,this.count=r}get astNodeType(){return"array"}get isArray(){return!0}}class ur extends K{constructor(e,t,n){super(e),this.format=t,this.access=n}get astNodeType(){return"sampler"}}class an extends Xt{constructor(){super(),this.postfix=null}}class ms extends an{constructor(e){super(),this.value=e}get astNodeType(){return"stringExpr"}toString(){return this.value}constEvaluateString(){return this.value}}class fn extends an{constructor(e,t){super(),this.type=e,this.args=t}get astNodeType(){return"createExpr"}search(e){if(e(this),this.args)for(const t of this.args)t.search(e)}constEvaluate(e,t){return t&&(t[0]=this.type),e.evalExpression(this,e.context)}}class ql extends an{constructor(e,t){super(),this.cachedReturnValue=null,this.name=e,this.args=t}get astNodeType(){return"callExpr"}setCachedReturnValue(e){this.cachedReturnValue=e}get isBuiltin(){return Op.has(this.name)}constEvaluate(e,t){return e.evalExpression(this,e.context)}search(e){for(const t of this.args)t.search(e);e(this)}}class Mt extends an{constructor(e){super(),this.name=e}get astNodeType(){return"varExpr"}search(e){e(this),this.postfix&&this.postfix.search(e)}constEvaluate(e,t){return e.evalExpression(this,e.context)}}class qp extends an{constructor(e,t){super(),this.name=e,this.initializer=t}get astNodeType(){return"constExpr"}constEvaluate(e,t){if(this.initializer){const n=e.evalExpression(this.initializer,e.context);return n!==null&&this.postfix?n.getSubData(e,this.postfix,e.context):n}return null}search(e){this.initializer.search(e)}}class We extends an{constructor(e,t){super(),this.value=e,this.type=t}get astNodeType(){return"literalExpr"}constEvaluate(e,t){return t!==void 0&&(t[0]=this.type),this.value}get isScalar(){return this.value instanceof L}get isVector(){return this.value instanceof M||this.value instanceof ue}get scalarValue(){return this.value instanceof L?this.value.value:(console.error("Value is not scalar."),0)}get vectorValue(){return this.value instanceof M||this.value instanceof ue?this.value.data:(console.error("Value is not a vector or matrix."),new Float32Array(0))}}class Hp extends an{constructor(e,t){super(),this.type=e,this.value=t}get astNodeType(){return"bitcastExpr"}search(e){this.value.search(e)}}class Ls extends an{constructor(e){super(),this.index=e}search(e){this.index.search(e)}}class jp extends an{constructor(){super()}}class Ue extends jp{constructor(e,t){super(),this.operator=e,this.right=t}get astNodeType(){return"unaryOp"}constEvaluate(e,t){return e.evalExpression(this,e.context)}search(e){this.right.search(e)}}class Jt extends jp{constructor(e,t,n){super(),this.operator=e,this.left=t,this.right=n}get astNodeType(){return"binaryOp"}_getPromotedType(e,t){return e.name===t.name?e:e.name==="f32"||t.name==="f32"?K.f32:e.name==="u32"||t.name==="u32"?K.u32:K.i32}constEvaluate(e,t){return e.evalExpression(this,e.context)}search(e){this.left.search(e),this.right.search(e)}}class Kp extends Xt{constructor(e){super(),this.body=e}search(e){e(this),this.searchBlock(this.body,e)}}class ki extends an{constructor(){super()}get astNodeType(){return"default"}}class Xp extends Kp{constructor(e,t){super(t),this.selectors=e}get astNodeType(){return"case"}search(e){this.searchBlock(this.body,e)}}class Yp extends Kp{constructor(e){super(e)}get astNodeType(){return"default"}search(e){this.searchBlock(this.body,e)}}class Rc extends Xt{constructor(e,t,n){super(),this.name=e,this.type=t,this.attributes=n}get astNodeType(){return"argument"}}class zk extends Xt{constructor(e,t){super(),this.condition=e,this.body=t}get astNodeType(){return"elseif"}search(e){this.condition.search(e),this.searchBlock(this.body,e)}}class Lc extends Xt{constructor(e,t,n){super(),this.name=e,this.type=t,this.attributes=n}get astNodeType(){return"member"}}class Qp extends Xt{constructor(e,t){super(),this.name=e,this.value=t}get astNodeType(){return"attribute"}}class qt{constructor(e,t){this.parent=null,this.typeInfo=e,this.parent=t,this.id=qt._id++}clone(){throw`Clone: Not implemented for ${this.constructor.name}`}setDataValue(e,t,n,r){console.error(`SetDataValue: Not implemented for ${this.constructor.name}`)}getSubData(e,t,n){return console.error(`GetDataValue: Not implemented for ${this.constructor.name}`),null}toString(){return`<${this.typeInfo.getTypeName()}>`}}qt._id=0;class Ba extends qt{constructor(){super(new Gt("void",null),null)}toString(){return"void"}}Ba.void=new Ba;class xs extends qt{constructor(e){super(new Ra("pointer",e.typeInfo,null),null),this.reference=e}clone(){return this}setDataValue(e,t,n,r){this.reference.setDataValue(e,t,n,r)}getSubData(e,t,n){return t?this.reference.getSubData(e,t,n):this}toString(){return`&${this.reference.toString()}`}}class L extends qt{constructor(e,t,n=null){super(t,n),e instanceof Int32Array||e instanceof Uint32Array||e instanceof Float32Array?this.data=e:this.typeInfo.name==="x32"?e-Math.floor(e)!==0?this.data=new Float32Array([e]):this.data=e>=0?new Uint32Array([e]):new Int32Array([e]):this.typeInfo.name==="i32"||this.typeInfo.name==="bool"?this.data=new Int32Array([e]):this.typeInfo.name==="u32"?this.data=new Uint32Array([e]):this.typeInfo.name==="f32"||this.typeInfo.name==="f16"?this.data=new Float32Array([e]):console.error("ScalarData2: Invalid type",t)}clone(){if(this.data instanceof Float32Array)return new L(new Float32Array(this.data),this.typeInfo,null);if(this.data instanceof Int32Array)return new L(new Int32Array(this.data),this.typeInfo,null);if(this.data instanceof Uint32Array)return new L(new Uint32Array(this.data),this.typeInfo,null);throw"ScalarData: Invalid data type"}get value(){return this.data[0]}set value(e){this.data[0]=e}setDataValue(e,t,n,r){if(n)return void console.error("SetDataValue: Scalar data does not support postfix",n);if(!(t instanceof L))return void console.error("SetDataValue: Invalid value",t);let i=t.data[0];this.typeInfo.name==="i32"||this.typeInfo.name==="u32"?i=Math.floor(i):this.typeInfo.name==="bool"&&(i=i?1:0),this.data[0]=i}getSubData(e,t,n){return t?(console.error("getSubData: Scalar data does not support postfix",t),null):this}toString(){return`${this.value}`}}function Vk(s,e,t){const n=e.length;return n===2?t==="f32"?new M(new Float32Array(e),s.getTypeInfo("vec2f")):t==="i32"||t==="bool"?new M(new Int32Array(e),s.getTypeInfo("vec2i")):t==="u32"?new M(new Uint32Array(e),s.getTypeInfo("vec2u")):t==="f16"?new M(new Float32Array(e),s.getTypeInfo("vec2h")):(console.error(`getSubData: Unknown format ${t}`),null):n===3?t==="f32"?new M(new Float32Array(e),s.getTypeInfo("vec3f")):t==="i32"||t==="bool"?new M(new Int32Array(e),s.getTypeInfo("vec3i")):t==="u32"?new M(new Uint32Array(e),s.getTypeInfo("vec3u")):t==="f16"?new M(new Float32Array(e),s.getTypeInfo("vec3h")):(console.error(`getSubData: Unknown format ${t}`),null):n===4?t==="f32"?new M(new Float32Array(e),s.getTypeInfo("vec4f")):t==="i32"||t==="bool"?new M(new Int32Array(e),s.getTypeInfo("vec4i")):t==="u32"?new M(new Uint32Array(e),s.getTypeInfo("vec4u")):t==="f16"?new M(new Float32Array(e),s.getTypeInfo("vec4h")):(console.error(`getSubData: Unknown format ${t}`),null):(console.error(`getSubData: Invalid vector size ${e.length}`),null)}class M extends qt{constructor(e,t,n=null){if(super(t,n),e instanceof Float32Array||e instanceof Uint32Array||e instanceof Int32Array)this.data=e;else{const r=this.typeInfo.name;r==="vec2f"||r==="vec3f"||r==="vec4f"?this.data=new Float32Array(e):r==="vec2i"||r==="vec3i"||r==="vec4i"?this.data=new Int32Array(e):r==="vec2u"||r==="vec3u"||r==="vec4u"?this.data=new Uint32Array(e):r==="vec2h"||r==="vec3h"||r==="vec4h"?this.data=new Float32Array(e):r==="vec2b"||r==="vec3b"||r==="vec4b"?this.data=new Int32Array(e):r==="vec2"||r==="vec3"||r==="vec4"?this.data=new Float32Array(e):console.error(`VectorData: Invalid type ${r}`)}}clone(){if(this.data instanceof Float32Array)return new M(new Float32Array(this.data),this.typeInfo,null);if(this.data instanceof Int32Array)return new M(new Int32Array(this.data),this.typeInfo,null);if(this.data instanceof Uint32Array)return new M(new Uint32Array(this.data),this.typeInfo,null);throw"VectorData: Invalid data type"}setDataValue(e,t,n,r){n instanceof ms?console.error("TODO: Set vector postfix"):t instanceof M?this.data=t.data:console.error("SetDataValue: Invalid value",t)}getSubData(e,t,n){if(t===null)return this;let r=e.getTypeInfo("f32");if(this.typeInfo instanceof ps)r=this.typeInfo.format||r;else{const o=this.typeInfo.name;o==="vec2f"||o==="vec3f"||o==="vec4f"?r=e.getTypeInfo("f32"):o==="vec2i"||o==="vec3i"||o==="vec4i"?r=e.getTypeInfo("i32"):o==="vec2b"||o==="vec3b"||o==="vec4b"?r=e.getTypeInfo("bool"):o==="vec2u"||o==="vec3u"||o==="vec4u"?r=e.getTypeInfo("u32"):o==="vec2h"||o==="vec3h"||o==="vec4h"?r=e.getTypeInfo("f16"):console.error(`GetSubData: Unknown type ${o}`)}let i=this;for(;t!==null&&i!==null;){if(t instanceof Ls){const o=t.index;let a=-1;if(o instanceof We){if(!(o.value instanceof L))return console.error(`GetSubData: Invalid array index ${o.value}`),null;a=o.value.value}else{const l=e.evalExpression(o,n);if(!(l instanceof L))return console.error("GetSubData: Unknown index type",o),null;a=l.value}if(a<0||a>=i.data.length)return console.error("GetSubData: Index out of range",a),null;if(i.data instanceof Float32Array){const l=new Float32Array(i.data.buffer,i.data.byteOffset+4*a,1);return new L(l,r)}if(i.data instanceof Int32Array){const l=new Int32Array(i.data.buffer,i.data.byteOffset+4*a,1);return new L(l,r)}if(i.data instanceof Uint32Array){const l=new Uint32Array(i.data.buffer,i.data.byteOffset+4*a,1);return new L(l,r)}throw"GetSubData: Invalid data type"}if(!(t instanceof ms))return console.error("GetSubData: Unknown postfix",t),null;{const o=t.value.toLowerCase();if(o.length===1){let l=0;if(o==="x"||o==="r")l=0;else if(o==="y"||o==="g")l=1;else if(o==="z"||o==="b")l=2;else{if(o!=="w"&&o!=="a")return console.error(`GetSubData: Unknown member ${o}`),null;l=3}if(this.data instanceof Float32Array){let u=new Float32Array(this.data.buffer,this.data.byteOffset+4*l,1);return new L(u,r,this)}if(this.data instanceof Int32Array){let u=new Int32Array(this.data.buffer,this.data.byteOffset+4*l,1);return new L(u,r,this)}if(this.data instanceof Uint32Array){let u=new Uint32Array(this.data.buffer,this.data.byteOffset+4*l,1);return new L(u,r,this)}}const a=[];for(const l of o)l==="x"||l==="r"?a.push(this.data[0]):l==="y"||l==="g"?a.push(this.data[1]):l==="z"||l==="b"?a.push(this.data[2]):l==="w"||l==="a"?a.push(this.data[3]):console.error(`GetDataValue: Unknown member ${l}`);i=Vk(e,a,r.name)}t=t.postfix}return i}toString(){let e=`${this.data[0]}`;for(let t=1;t<this.data.length;++t)e+=`, ${this.data[t]}`;return e}}class ue extends qt{constructor(e,t,n=null){super(t,n),e instanceof Float32Array?this.data=e:this.data=new Float32Array(e)}clone(){return new ue(new Float32Array(this.data),this.typeInfo,null)}setDataValue(e,t,n,r){n instanceof ms?console.error("TODO: Set matrix postfix"):t instanceof ue?this.data=t.data:console.error("SetDataValue: Invalid value",t)}getSubData(e,t,n){if(t===null)return this;const r=this.typeInfo.name;if(e.getTypeInfo("f32"),this.typeInfo instanceof ps)this.typeInfo.format;else if(r.endsWith("f"))e.getTypeInfo("f32");else if(r.endsWith("i"))e.getTypeInfo("i32");else if(r.endsWith("u"))e.getTypeInfo("u32");else{if(!r.endsWith("h"))return console.error(`GetDataValue: Unknown type ${r}`),null;e.getTypeInfo("f16")}if(t instanceof Ls){const i=t.index;let o=-1;if(i instanceof We){if(!(i.value instanceof L))return console.error(`GetDataValue: Invalid array index ${i.value}`),null;o=i.value.value}else{const u=e.evalExpression(i,n);if(!(u instanceof L))return console.error("GetDataValue: Unknown index type",i),null;o=u.value}if(o<0||o>=this.data.length)return console.error("GetDataValue: Index out of range",o),null;const a=r.endsWith("h")?"h":"f";let l;if(r==="mat2x2"||r==="mat2x2f"||r==="mat2x2h"||r==="mat3x2"||r==="mat3x2f"||r==="mat3x2h"||r==="mat4x2"||r==="mat4x2f"||r==="mat4x2h")l=new M(new Float32Array(this.data.buffer,this.data.byteOffset+8*o,2),e.getTypeInfo(`vec2${a}`));else if(r==="mat2x3"||r==="mat2x3f"||r==="mat2x3h"||r==="mat3x3"||r==="mat3x3f"||r==="mat3x3h"||r==="mat4x3"||r==="mat4x3f"||r==="mat4x3h")l=new M(new Float32Array(this.data.buffer,this.data.byteOffset+12*o,3),e.getTypeInfo(`vec3${a}`));else{if(r!=="mat2x4"&&r!=="mat2x4f"&&r!=="mat2x4h"&&r!=="mat3x4"&&r!=="mat3x4f"&&r!=="mat3x4h"&&r!=="mat4x4"&&r!=="mat4x4f"&&r!=="mat4x4h")return console.error(`GetDataValue: Unknown type ${r}`),null;l=new M(new Float32Array(this.data.buffer,this.data.byteOffset+16*o,4),e.getTypeInfo(`vec4${a}`))}return t.postfix?l.getSubData(e,t.postfix,n):l}return console.error("GetDataValue: Invalid postfix",t),null}toString(){let e=`${this.data[0]}`;for(let t=1;t<this.data.length;++t)e+=`, ${this.data[t]}`;return e}}class Le extends qt{constructor(e,t,n=0,r=null){super(t,r),this.buffer=e instanceof ArrayBuffer?e:e.buffer,this.offset=n}clone(){const e=new Uint8Array(new Uint8Array(this.buffer,this.offset,this.typeInfo.size));return new Le(e.buffer,this.typeInfo,0,null)}setDataValue(e,t,n,r){if(t===null)return void console.log("setDataValue: NULL data.");let i=this.offset,o=this.typeInfo;for(;n;){if(n instanceof Ls)if(o instanceof Vn){const a=n.index;if(a instanceof We){if(!(a.value instanceof L))return void console.error(`SetDataValue: Invalid index type ${a.value}`);i+=a.value.value*o.stride}else{const l=e.evalExpression(a,r);if(!(l instanceof L))return void console.error("SetDataValue: Unknown index type",a);i+=l.value*o.stride}o=o.format}else console.error(`SetDataValue: Type ${o.getTypeName()} is not an array`);else{if(!(n instanceof ms))return void console.error("SetDataValue: Unknown postfix type",n);{const a=n.value;if(o instanceof Ln){let l=!1;for(const u of o.members)if(u.name===a){i+=u.offset,o=u.type,l=!0;break}if(!l)return void console.error(`SetDataValue: Member ${a} not found`)}else if(o instanceof Gt){const l=o.getTypeName();let u=0;if(a==="x"||a==="r")u=0;else if(a==="y"||a==="g")u=1;else if(a==="z"||a==="b")u=2;else{if(a!=="w"&&a!=="a")return void console.error(`SetDataValue: Unknown member ${a}`);u=3}if(!(t instanceof L))return void console.error("SetDataValue: Invalid value",t);const c=t.value;return l==="vec2f"?void(new Float32Array(this.buffer,i,2)[u]=c):l==="vec3f"?void(new Float32Array(this.buffer,i,3)[u]=c):l==="vec4f"?void(new Float32Array(this.buffer,i,4)[u]=c):l==="vec2i"?void(new Int32Array(this.buffer,i,2)[u]=c):l==="vec3i"?void(new Int32Array(this.buffer,i,3)[u]=c):l==="vec4i"?void(new Int32Array(this.buffer,i,4)[u]=c):l==="vec2u"?void(new Uint32Array(this.buffer,i,2)[u]=c):l==="vec3u"?void(new Uint32Array(this.buffer,i,3)[u]=c):l==="vec4u"?void(new Uint32Array(this.buffer,i,4)[u]=c):void console.error(`SetDataValue: Type ${l} is not a struct`)}}}n=n.postfix}this.setData(e,t,o,i,r)}setData(e,t,n,r,i){const o=n.getTypeName();if(o!=="f32"&&o!=="f16")if(o!=="i32"&&o!=="atomic<i32>"&&o!=="x32")if(o!=="u32"&&o!=="atomic<u32>")if(o!=="bool"){if(o==="vec2f"||o==="vec2h"){const a=new Float32Array(this.buffer,r,2);return void(t instanceof M?(a[0]=t.data[0],a[1]=t.data[1]):(a[0]=t[0],a[1]=t[1]))}if(o==="vec3f"||o==="vec3h"){const a=new Float32Array(this.buffer,r,3);return void(t instanceof M?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2]):(a[0]=t[0],a[1]=t[1],a[2]=t[2]))}if(o==="vec4f"||o==="vec4h"){const a=new Float32Array(this.buffer,r,4);return void(t instanceof M?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3]))}if(o==="vec2i"){const a=new Int32Array(this.buffer,r,2);return void(t instanceof M?(a[0]=t.data[0],a[1]=t.data[1]):(a[0]=t[0],a[1]=t[1]))}if(o==="vec3i"){const a=new Int32Array(this.buffer,r,3);return void(t instanceof M?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2]):(a[0]=t[0],a[1]=t[1],a[2]=t[2]))}if(o==="vec4i"){const a=new Int32Array(this.buffer,r,4);return void(t instanceof M?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3]))}if(o==="vec2u"){const a=new Uint32Array(this.buffer,r,2);return void(t instanceof M?(a[0]=t.data[0],a[1]=t.data[1]):(a[0]=t[0],a[1]=t[1]))}if(o==="vec3u"){const a=new Uint32Array(this.buffer,r,3);return void(t instanceof M?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2]):(a[0]=t[0],a[1]=t[1],a[2]=t[2]))}if(o==="vec4u"){const a=new Uint32Array(this.buffer,r,4);return void(t instanceof M?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3]))}if(o==="vec2b"){const a=new Uint32Array(this.buffer,r,2);return void(t instanceof M?(a[0]=t.data[0],a[1]=t.data[1]):(a[0]=t[0],a[1]=t[1]))}if(o==="vec3b"){const a=new Uint32Array(this.buffer,r,3);return void(t instanceof M?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2]):(a[0]=t[0],a[1]=t[1],a[2]=t[2]))}if(o==="vec4b"){const a=new Uint32Array(this.buffer,r,4);return void(t instanceof M?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3]))}if(o==="mat2x2f"||o==="mat2x2h"){const a=new Float32Array(this.buffer,r,4);return void(t instanceof ue?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3]))}if(o==="mat2x3f"||o==="mat2x3h"){const a=new Float32Array(this.buffer,r,6);return void(t instanceof ue?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3],a[4]=t.data[4],a[5]=t.data[5]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3],a[4]=t[4],a[5]=t[5]))}if(o==="mat2x4f"||o==="mat2x4h"){const a=new Float32Array(this.buffer,r,8);return void(t instanceof ue?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3],a[4]=t.data[4],a[5]=t.data[5],a[6]=t.data[6],a[7]=t.data[7]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3],a[4]=t[4],a[5]=t[5],a[6]=t[6],a[7]=t[7]))}if(o==="mat3x2f"||o==="mat3x2h"){const a=new Float32Array(this.buffer,r,6);return void(t instanceof ue?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3],a[4]=t.data[4],a[5]=t.data[5]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3],a[4]=t[4],a[5]=t[5]))}if(o==="mat3x3f"||o==="mat3x3h"){const a=new Float32Array(this.buffer,r,9);return void(t instanceof ue?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3],a[4]=t.data[4],a[5]=t.data[5],a[6]=t.data[6],a[7]=t.data[7],a[8]=t.data[8]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3],a[4]=t[4],a[5]=t[5],a[6]=t[6],a[7]=t[7],a[8]=t[8]))}if(o==="mat3x4f"||o==="mat3x4h"){const a=new Float32Array(this.buffer,r,12);return void(t instanceof ue?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3],a[4]=t.data[4],a[5]=t.data[5],a[6]=t.data[6],a[7]=t.data[7],a[8]=t.data[8],a[9]=t.data[9],a[10]=t.data[10],a[11]=t.data[11]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3],a[4]=t[4],a[5]=t[5],a[6]=t[6],a[7]=t[7],a[8]=t[8],a[9]=t[9],a[10]=t[10],a[11]=t[11]))}if(o==="mat4x2f"||o==="mat4x2h"){const a=new Float32Array(this.buffer,r,8);return void(t instanceof ue?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3],a[4]=t.data[4],a[5]=t.data[5],a[6]=t.data[6],a[7]=t.data[7]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3],a[4]=t[4],a[5]=t[5],a[6]=t[6],a[7]=t[7]))}if(o==="mat4x3f"||o==="mat4x3h"){const a=new Float32Array(this.buffer,r,12);return void(t instanceof ue?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3],a[4]=t.data[4],a[5]=t.data[5],a[6]=t.data[6],a[7]=t.data[7],a[8]=t.data[8],a[9]=t.data[9],a[10]=t.data[10],a[11]=t.data[11]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3],a[4]=t[4],a[5]=t[5],a[6]=t[6],a[7]=t[7],a[8]=t[8],a[9]=t[9],a[10]=t[10],a[11]=t[11]))}if(o==="mat4x4f"||o==="mat4x4h"){const a=new Float32Array(this.buffer,r,16);return void(t instanceof ue?(a[0]=t.data[0],a[1]=t.data[1],a[2]=t.data[2],a[3]=t.data[3],a[4]=t.data[4],a[5]=t.data[5],a[6]=t.data[6],a[7]=t.data[7],a[8]=t.data[8],a[9]=t.data[9],a[10]=t.data[10],a[11]=t.data[11],a[12]=t.data[12],a[13]=t.data[13],a[14]=t.data[14],a[15]=t.data[15]):(a[0]=t[0],a[1]=t[1],a[2]=t[2],a[3]=t[3],a[4]=t[4],a[5]=t[5],a[6]=t[6],a[7]=t[7],a[8]=t[8],a[9]=t[9],a[10]=t[10],a[11]=t[11],a[12]=t[12],a[13]=t[13],a[14]=t[14],a[15]=t[15]))}if(t instanceof Le){if(n===t.typeInfo)return void new Uint8Array(this.buffer,r,t.buffer.byteLength).set(new Uint8Array(t.buffer));console.error("SetDataValue: Type mismatch",o,t.typeInfo.getTypeName())}else console.error(`SetData: Unknown type ${o}`)}else t instanceof L&&(new Int32Array(this.buffer,r,1)[0]=t.value);else t instanceof L&&(new Uint32Array(this.buffer,r,1)[0]=t.value);else t instanceof L&&(new Int32Array(this.buffer,r,1)[0]=t.value);else t instanceof L&&(new Float32Array(this.buffer,r,1)[0]=t.value)}getSubData(e,t,n){var r,i,o;if(t===null)return this;let a=this.offset,l=this.typeInfo;for(;t;){if(t instanceof Ls){const c=t.index,h=c instanceof an?e.evalExpression(c,n):c;let d=0;if(h instanceof L?d=h.value:typeof h=="number"?d=h:console.error("GetDataValue: Invalid index type",c),l instanceof Vn)a+=d*l.stride,l=l.format;else{const w=l.getTypeName();w==="mat4x4"||w==="mat4x4f"||w==="mat4x4h"?(a+=16*d,l=e.getTypeInfo("vec4f")):console.error(`getDataValue: Type ${l.getTypeName()} is not an array`)}}else{if(!(t instanceof ms))return console.error("GetDataValue: Unknown postfix type",t),null;{const c=t.value;if(l instanceof Ln){let h=!1;for(const d of l.members)if(d.name===c){a+=d.offset,l=d.type,h=!0;break}if(!h)return console.error(`GetDataValue: Member ${c} not found`),null}else if(l instanceof Gt){const h=l.getTypeName();if(h==="vec2f"||h==="vec3f"||h==="vec4f"||h==="vec2i"||h==="vec3i"||h==="vec4i"||h==="vec2u"||h==="vec3u"||h==="vec4u"||h==="vec2b"||h==="vec3b"||h==="vec4b"||h==="vec2h"||h==="vec3h"||h==="vec4h"||h==="vec2"||h==="vec3"||h==="vec4"){if(c.length>0&&c.length<5){let d="f";const w=[];for(let I=0;I<c.length;++I){const E=c[I].toLowerCase();let m=0;if(E==="x"||E==="r")m=0;else if(E==="y"||E==="g")m=1;else if(E==="z"||E==="b")m=2;else{if(E!=="w"&&E!=="a")return console.error(`Unknown member ${c}`),null;m=3}if(c.length===1){if(h.endsWith("f"))return this.buffer.byteLength<a+4*m+4?(console.log("Insufficient buffer data"),null):new L(new Float32Array(this.buffer,a+4*m,1),e.getTypeInfo("f32"),this);if(h.endsWith("h"))return new L(new Float32Array(this.buffer,a+4*m,1),e.getTypeInfo("f16"),this);if(h.endsWith("i"))return new L(new Int32Array(this.buffer,a+4*m,1),e.getTypeInfo("i32"),this);if(h.endsWith("b"))return new L(new Int32Array(this.buffer,a+4*m,1),e.getTypeInfo("bool"),this);if(h.endsWith("u"))return new L(new Uint32Array(this.buffer,a+4*m,1),e.getTypeInfo("i32"),this)}if(h==="vec2f")w.push(new Float32Array(this.buffer,a,2)[m]);else if(h==="vec3f"){if(a+12>=this.buffer.byteLength)return console.log("Insufficient buffer data"),null;const S=new Float32Array(this.buffer,a,3);w.push(S[m])}else if(h==="vec4f")w.push(new Float32Array(this.buffer,a,4)[m]);else if(h==="vec2i")d="i",w.push(new Int32Array(this.buffer,a,2)[m]);else if(h==="vec3i")d="i",w.push(new Int32Array(this.buffer,a,3)[m]);else if(h==="vec4i")d="i",w.push(new Int32Array(this.buffer,a,4)[m]);else if(h==="vec2u"){d="u";const S=new Uint32Array(this.buffer,a,2);w.push(S[m])}else h==="vec3u"?(d="u",w.push(new Uint32Array(this.buffer,a,3)[m])):h==="vec4u"&&(d="u",w.push(new Uint32Array(this.buffer,a,4)[m]))}return w.length===2?l=e.getTypeInfo(`vec2${d}`):w.length===3?l=e.getTypeInfo(`vec3${d}`):w.length===4?l=e.getTypeInfo(`vec4${d}`):console.error(`GetDataValue: Invalid vector length ${w.length}`),new M(w,l,null)}return console.error(`GetDataValue: Unknown member ${c}`),null}return console.error(`GetDataValue: Type ${h} is not a struct`),null}}}t=t.postfix}const u=l.getTypeName();return u==="f32"?new L(new Float32Array(this.buffer,a,1),l,this):u==="i32"?new L(new Int32Array(this.buffer,a,1),l,this):u==="u32"?new L(new Uint32Array(this.buffer,a,1),l,this):u==="vec2f"?new M(new Float32Array(this.buffer,a,2),l,this):u==="vec3f"?new M(new Float32Array(this.buffer,a,3),l,this):u==="vec4f"?new M(new Float32Array(this.buffer,a,4),l,this):u==="vec2i"?new M(new Int32Array(this.buffer,a,2),l,this):u==="vec3i"?new M(new Int32Array(this.buffer,a,3),l,this):u==="vec4i"?new M(new Int32Array(this.buffer,a,4),l,this):u==="vec2u"?new M(new Uint32Array(this.buffer,a,2),l,this):u==="vec3u"?new M(new Uint32Array(this.buffer,a,3),l,this):u==="vec4u"?new M(new Uint32Array(this.buffer,a,4),l,this):l instanceof ps&&l.name==="atomic"?((r=l.format)===null||r===void 0?void 0:r.name)==="u32"?new L(new Uint32Array(this.buffer,a,1)[0],l.format,this):((i=l.format)===null||i===void 0?void 0:i.name)==="i32"?new L(new Int32Array(this.buffer,a,1)[0],l.format,this):(console.error(`GetDataValue: Invalid atomic format ${(o=l.format)===null||o===void 0?void 0:o.name}`),null):new Le(this.buffer,l,a,this)}toString(){let e="";if(this.typeInfo instanceof Vn)if(this.typeInfo.format.name==="f32"){const t=new Float32Array(this.buffer,this.offset);e=`[${t[0]}`;for(let n=1;n<t.length;++n)e+=`, ${t[n]}`}else if(this.typeInfo.format.name==="i32"){const t=new Int32Array(this.buffer,this.offset);e=`[${t[0]}`;for(let n=1;n<t.length;++n)e+=`, ${t[n]}`}else if(this.typeInfo.format.name==="u32"){const t=new Uint32Array(this.buffer,this.offset);e=`[${t[0]}`;for(let n=1;n<t.length;++n)e+=`, ${t[n]}`}else if(this.typeInfo.format.name==="vec2f"){const t=new Float32Array(this.buffer,this.offset);e=`[${t[0]}, ${t[1]}]`;for(let n=1;n<t.length/2;++n)e+=`, [${t[2*n]}, ${t[2*n+1]}]`}else if(this.typeInfo.format.name==="vec3f"){const t=new Float32Array(this.buffer,this.offset);e=`[${t[0]}, ${t[1]}, ${t[2]}]`;for(let n=4;n<t.length;n+=4)e+=`, [${t[n]}, ${t[n+1]}, ${t[n+2]}]`}else if(this.typeInfo.format.name==="vec4f"){const t=new Float32Array(this.buffer,this.offset);e=`[${t[0]}, ${t[1]}, ${t[2]}, ${t[3]}]`;for(let n=4;n<t.length;n+=4)e+=`, [${t[n]}, ${t[n+1]}, ${t[n+2]}, ${t[n+3]}]`}else e="[...]";else this.typeInfo instanceof Ln?e+="{...}":e="[...]";return e}}class Tn extends qt{constructor(e,t,n,r){super(t,null),this.data=e,this.descriptor=n,this.view=r}clone(){return new Tn(this.data,this.typeInfo,this.descriptor,this.view)}get width(){var e,t;const n=this.descriptor.size;return n instanceof Array&&n.length>0?(e=n[0])!==null&&e!==void 0?e:0:n instanceof Object&&(t=n.width)!==null&&t!==void 0?t:0}get height(){var e,t;const n=this.descriptor.size;return n instanceof Array&&n.length>1?(e=n[1])!==null&&e!==void 0?e:0:n instanceof Object&&(t=n.height)!==null&&t!==void 0?t:0}get depthOrArrayLayers(){var e,t;const n=this.descriptor.size;return n instanceof Array&&n.length>2?(e=n[2])!==null&&e!==void 0?e:0:n instanceof Object&&(t=n.depthOrArrayLayers)!==null&&t!==void 0?t:0}get format(){var e;return this.descriptor&&(e=this.descriptor.format)!==null&&e!==void 0?e:"rgba8unorm"}get sampleCount(){var e;return this.descriptor&&(e=this.descriptor.sampleCount)!==null&&e!==void 0?e:1}get mipLevelCount(){var e;return this.descriptor&&(e=this.descriptor.mipLevelCount)!==null&&e!==void 0?e:1}get dimension(){var e;return this.descriptor&&(e=this.descriptor.dimension)!==null&&e!==void 0?e:"2d"}getMipLevelSize(e){if(e>=this.mipLevelCount)return[0,0,0];const t=[this.width,this.height,this.depthOrArrayLayers];for(let n=0;n<t.length;++n)t[n]=Math.max(1,t[n]>>e);return t}get texelByteSize(){const e=this.format,t=ta[e];return t?t.isDepthStencil?4:t.bytesPerBlock:0}get bytesPerRow(){return this.width*this.texelByteSize}get isDepthStencil(){const e=this.format,t=ta[e];return!!t&&t.isDepthStencil}getGpuSize(){const e=this.format,t=ta[e],n=this.width;if(!e||n<=0||!t)return-1;const r=this.height,i=this.depthOrArrayLayers,o=this.dimension;return n/t.blockWidth*(o==="1d"?1:r/t.blockHeight)*t.bytesPerBlock*i}getPixel(e,t,n=0,r=0){const i=this.texelByteSize,o=this.bytesPerRow,a=this.height,l=this.data[r];return((u,c,h,d,w,I,E,m)=>{const S=d*(E>>=w)*(I>>=w)+h*E+c*m;switch(this.format){case"r8unorm":return[ye(u,S,"8unorm",1)[0]];case"r8snorm":return[ye(u,S,"8snorm",1)[0]];case"r8uint":return[ye(u,S,"8uint",1)[0]];case"r8sint":return[ye(u,S,"8sint",1)[0]];case"rg8unorm":{const b=ye(u,S,"8unorm",2);return[b[0],b[1]]}case"rg8snorm":{const b=ye(u,S,"8snorm",2);return[b[0],b[1]]}case"rg8uint":{const b=ye(u,S,"8uint",2);return[b[0],b[1]]}case"rg8sint":{const b=ye(u,S,"8sint",2);return[b[0],b[1]]}case"rgba8unorm-srgb":case"rgba8unorm":{const b=ye(u,S,"8unorm",4);return[b[0],b[1],b[2],b[3]]}case"rgba8snorm":{const b=ye(u,S,"8snorm",4);return[b[0],b[1],b[2],b[3]]}case"rgba8uint":{const b=ye(u,S,"8uint",4);return[b[0],b[1],b[2],b[3]]}case"rgba8sint":{const b=ye(u,S,"8sint",4);return[b[0],b[1],b[2],b[3]]}case"bgra8unorm-srgb":case"bgra8unorm":{const b=ye(u,S,"8unorm",4);return[b[2],b[1],b[0],b[3]]}case"r16uint":return[ye(u,S,"16uint",1)[0]];case"r16sint":return[ye(u,S,"16sint",1)[0]];case"r16float":return[ye(u,S,"16float",1)[0]];case"rg16uint":{const b=ye(u,S,"16uint",2);return[b[0],b[1]]}case"rg16sint":{const b=ye(u,S,"16sint",2);return[b[0],b[1]]}case"rg16float":{const b=ye(u,S,"16float",2);return[b[0],b[1]]}case"rgba16uint":{const b=ye(u,S,"16uint",4);return[b[0],b[1],b[2],b[3]]}case"rgba16sint":{const b=ye(u,S,"16sint",4);return[b[0],b[1],b[2],b[3]]}case"rgba16float":{const b=ye(u,S,"16float",4);return[b[0],b[1],b[2],b[3]]}case"r32uint":return[ye(u,S,"32uint",1)[0]];case"r32sint":return[ye(u,S,"32sint",1)[0]];case"depth16unorm":case"depth24plus":case"depth24plus-stencil8":case"depth32float":case"depth32float-stencil8":case"r32float":return[ye(u,S,"32float",1)[0]];case"rg32uint":{const b=ye(u,S,"32uint",2);return[b[0],b[1]]}case"rg32sint":{const b=ye(u,S,"32sint",2);return[b[0],b[1]]}case"rg32float":{const b=ye(u,S,"32float",2);return[b[0],b[1]]}case"rgba32uint":{const b=ye(u,S,"32uint",4);return[b[0],b[1],b[2],b[3]]}case"rgba32sint":{const b=ye(u,S,"32sint",4);return[b[0],b[1],b[2],b[3]]}case"rgba32float":{const b=ye(u,S,"32float",4);return[b[0],b[1],b[2],b[3]]}case"rg11b10ufloat":{const b=new Uint32Array(u.buffer,S,1)[0],f=(4192256&b)>>11,_=(4290772992&b)>>22;return[Mc(2047&b),Mc(f),(v=>{const T=112+(v>>5&31)<<23|(31&v)<<18;return zl[0]=T,Dp[0]})(_),1]}}return null})(new Uint8Array(l),e,t,n,r,a,o,i)}setPixel(e,t,n,r,i){const o=this.texelByteSize,a=this.bytesPerRow,l=this.height,u=this.data[r];((c,h,d,w,I,E,m,S,b,f)=>{const _=w*(m>>=I)*(E>>=I)+d*m+h*S;switch(b){case"r8unorm":return void ve(c,_,"8unorm",1,f);case"r8snorm":return void ve(c,_,"8snorm",1,f);case"r8uint":return void ve(c,_,"8uint",1,f);case"r8sint":return void ve(c,_,"8sint",1,f);case"rg8unorm":return void ve(c,_,"8unorm",2,f);case"rg8snorm":return void ve(c,_,"8snorm",2,f);case"rg8uint":return void ve(c,_,"8uint",2,f);case"rg8sint":return void ve(c,_,"8sint",2,f);case"rgba8unorm-srgb":case"rgba8unorm":case"bgra8unorm-srgb":case"bgra8unorm":return void ve(c,_,"8unorm",4,f);case"rgba8snorm":return void ve(c,_,"8snorm",4,f);case"rgba8uint":return void ve(c,_,"8uint",4,f);case"rgba8sint":return void ve(c,_,"8sint",4,f);case"r16uint":return void ve(c,_,"16uint",1,f);case"r16sint":return void ve(c,_,"16sint",1,f);case"r16float":return void ve(c,_,"16float",1,f);case"rg16uint":return void ve(c,_,"16uint",2,f);case"rg16sint":return void ve(c,_,"16sint",2,f);case"rg16float":return void ve(c,_,"16float",2,f);case"rgba16uint":return void ve(c,_,"16uint",4,f);case"rgba16sint":return void ve(c,_,"16sint",4,f);case"rgba16float":return void ve(c,_,"16float",4,f);case"r32uint":return void ve(c,_,"32uint",1,f);case"r32sint":return void ve(c,_,"32sint",1,f);case"depth16unorm":case"depth24plus":case"depth24plus-stencil8":case"depth32float":case"depth32float-stencil8":case"r32float":return void ve(c,_,"32float",1,f);case"rg32uint":return void ve(c,_,"32uint",2,f);case"rg32sint":return void ve(c,_,"32sint",2,f);case"rg32float":return void ve(c,_,"32float",2,f);case"rgba32uint":return void ve(c,_,"32uint",4,f);case"rgba32sint":return void ve(c,_,"32sint",4,f);case"rgba32float":return void ve(c,_,"32float",4,f);case"rg11b10ufloat":console.error("TODO: rg11b10ufloat not supported for writing")}})(new Uint8Array(u),e,t,n,r,l,a,o,this.format,i)}}(s=>{s[s.token=0]="token",s[s.keyword=1]="keyword",s[s.reserved=2]="reserved"})(B||(B={}));class F{constructor(e,t,n){this.name=e,this.type=t,this.rule=n}toString(){return this.name}}class D{}V=D,D.none=new F("",B.reserved,""),D.eof=new F("EOF",B.token,""),D.reserved={asm:new F("asm",B.reserved,"asm"),bf16:new F("bf16",B.reserved,"bf16"),do:new F("do",B.reserved,"do"),enum:new F("enum",B.reserved,"enum"),f16:new F("f16",B.reserved,"f16"),f64:new F("f64",B.reserved,"f64"),handle:new F("handle",B.reserved,"handle"),i8:new F("i8",B.reserved,"i8"),i16:new F("i16",B.reserved,"i16"),i64:new F("i64",B.reserved,"i64"),mat:new F("mat",B.reserved,"mat"),premerge:new F("premerge",B.reserved,"premerge"),regardless:new F("regardless",B.reserved,"regardless"),typedef:new F("typedef",B.reserved,"typedef"),u8:new F("u8",B.reserved,"u8"),u16:new F("u16",B.reserved,"u16"),u64:new F("u64",B.reserved,"u64"),unless:new F("unless",B.reserved,"unless"),using:new F("using",B.reserved,"using"),vec:new F("vec",B.reserved,"vec"),void:new F("void",B.reserved,"void")},D.keywords={array:new F("array",B.keyword,"array"),atomic:new F("atomic",B.keyword,"atomic"),bool:new F("bool",B.keyword,"bool"),f32:new F("f32",B.keyword,"f32"),i32:new F("i32",B.keyword,"i32"),mat2x2:new F("mat2x2",B.keyword,"mat2x2"),mat2x3:new F("mat2x3",B.keyword,"mat2x3"),mat2x4:new F("mat2x4",B.keyword,"mat2x4"),mat3x2:new F("mat3x2",B.keyword,"mat3x2"),mat3x3:new F("mat3x3",B.keyword,"mat3x3"),mat3x4:new F("mat3x4",B.keyword,"mat3x4"),mat4x2:new F("mat4x2",B.keyword,"mat4x2"),mat4x3:new F("mat4x3",B.keyword,"mat4x3"),mat4x4:new F("mat4x4",B.keyword,"mat4x4"),ptr:new F("ptr",B.keyword,"ptr"),sampler:new F("sampler",B.keyword,"sampler"),sampler_comparison:new F("sampler_comparison",B.keyword,"sampler_comparison"),struct:new F("struct",B.keyword,"struct"),texture_1d:new F("texture_1d",B.keyword,"texture_1d"),texture_2d:new F("texture_2d",B.keyword,"texture_2d"),texture_2d_array:new F("texture_2d_array",B.keyword,"texture_2d_array"),texture_3d:new F("texture_3d",B.keyword,"texture_3d"),texture_cube:new F("texture_cube",B.keyword,"texture_cube"),texture_cube_array:new F("texture_cube_array",B.keyword,"texture_cube_array"),texture_multisampled_2d:new F("texture_multisampled_2d",B.keyword,"texture_multisampled_2d"),texture_storage_1d:new F("texture_storage_1d",B.keyword,"texture_storage_1d"),texture_storage_2d:new F("texture_storage_2d",B.keyword,"texture_storage_2d"),texture_storage_2d_array:new F("texture_storage_2d_array",B.keyword,"texture_storage_2d_array"),texture_storage_3d:new F("texture_storage_3d",B.keyword,"texture_storage_3d"),texture_depth_2d:new F("texture_depth_2d",B.keyword,"texture_depth_2d"),texture_depth_2d_array:new F("texture_depth_2d_array",B.keyword,"texture_depth_2d_array"),texture_depth_cube:new F("texture_depth_cube",B.keyword,"texture_depth_cube"),texture_depth_cube_array:new F("texture_depth_cube_array",B.keyword,"texture_depth_cube_array"),texture_depth_multisampled_2d:new F("texture_depth_multisampled_2d",B.keyword,"texture_depth_multisampled_2d"),texture_external:new F("texture_external",B.keyword,"texture_external"),u32:new F("u32",B.keyword,"u32"),vec2:new F("vec2",B.keyword,"vec2"),vec3:new F("vec3",B.keyword,"vec3"),vec4:new F("vec4",B.keyword,"vec4"),bitcast:new F("bitcast",B.keyword,"bitcast"),block:new F("block",B.keyword,"block"),break:new F("break",B.keyword,"break"),case:new F("case",B.keyword,"case"),continue:new F("continue",B.keyword,"continue"),continuing:new F("continuing",B.keyword,"continuing"),default:new F("default",B.keyword,"default"),diagnostic:new F("diagnostic",B.keyword,"diagnostic"),discard:new F("discard",B.keyword,"discard"),else:new F("else",B.keyword,"else"),enable:new F("enable",B.keyword,"enable"),fallthrough:new F("fallthrough",B.keyword,"fallthrough"),false:new F("false",B.keyword,"false"),fn:new F("fn",B.keyword,"fn"),for:new F("for",B.keyword,"for"),function:new F("function",B.keyword,"function"),if:new F("if",B.keyword,"if"),let:new F("let",B.keyword,"let"),const:new F("const",B.keyword,"const"),loop:new F("loop",B.keyword,"loop"),while:new F("while",B.keyword,"while"),private:new F("private",B.keyword,"private"),read:new F("read",B.keyword,"read"),read_write:new F("read_write",B.keyword,"read_write"),return:new F("return",B.keyword,"return"),requires:new F("requires",B.keyword,"requires"),storage:new F("storage",B.keyword,"storage"),switch:new F("switch",B.keyword,"switch"),true:new F("true",B.keyword,"true"),alias:new F("alias",B.keyword,"alias"),type:new F("type",B.keyword,"type"),uniform:new F("uniform",B.keyword,"uniform"),var:new F("var",B.keyword,"var"),override:new F("override",B.keyword,"override"),workgroup:new F("workgroup",B.keyword,"workgroup"),write:new F("write",B.keyword,"write"),r8unorm:new F("r8unorm",B.keyword,"r8unorm"),r8snorm:new F("r8snorm",B.keyword,"r8snorm"),r8uint:new F("r8uint",B.keyword,"r8uint"),r8sint:new F("r8sint",B.keyword,"r8sint"),r16uint:new F("r16uint",B.keyword,"r16uint"),r16sint:new F("r16sint",B.keyword,"r16sint"),r16float:new F("r16float",B.keyword,"r16float"),rg8unorm:new F("rg8unorm",B.keyword,"rg8unorm"),rg8snorm:new F("rg8snorm",B.keyword,"rg8snorm"),rg8uint:new F("rg8uint",B.keyword,"rg8uint"),rg8sint:new F("rg8sint",B.keyword,"rg8sint"),r32uint:new F("r32uint",B.keyword,"r32uint"),r32sint:new F("r32sint",B.keyword,"r32sint"),r32float:new F("r32float",B.keyword,"r32float"),rg16uint:new F("rg16uint",B.keyword,"rg16uint"),rg16sint:new F("rg16sint",B.keyword,"rg16sint"),rg16float:new F("rg16float",B.keyword,"rg16float"),rgba8unorm:new F("rgba8unorm",B.keyword,"rgba8unorm"),rgba8unorm_srgb:new F("rgba8unorm_srgb",B.keyword,"rgba8unorm_srgb"),rgba8snorm:new F("rgba8snorm",B.keyword,"rgba8snorm"),rgba8uint:new F("rgba8uint",B.keyword,"rgba8uint"),rgba8sint:new F("rgba8sint",B.keyword,"rgba8sint"),bgra8unorm:new F("bgra8unorm",B.keyword,"bgra8unorm"),bgra8unorm_srgb:new F("bgra8unorm_srgb",B.keyword,"bgra8unorm_srgb"),rgb10a2unorm:new F("rgb10a2unorm",B.keyword,"rgb10a2unorm"),rg11b10float:new F("rg11b10float",B.keyword,"rg11b10float"),rg32uint:new F("rg32uint",B.keyword,"rg32uint"),rg32sint:new F("rg32sint",B.keyword,"rg32sint"),rg32float:new F("rg32float",B.keyword,"rg32float"),rgba16uint:new F("rgba16uint",B.keyword,"rgba16uint"),rgba16sint:new F("rgba16sint",B.keyword,"rgba16sint"),rgba16float:new F("rgba16float",B.keyword,"rgba16float"),rgba32uint:new F("rgba32uint",B.keyword,"rgba32uint"),rgba32sint:new F("rgba32sint",B.keyword,"rgba32sint"),rgba32float:new F("rgba32float",B.keyword,"rgba32float"),static_assert:new F("static_assert",B.keyword,"static_assert")},D.tokens={decimal_float_literal:new F("decimal_float_literal",B.token,/((-?[0-9]*\.[0-9]+|-?[0-9]+\.[0-9]*)((e|E)(\+|-)?[0-9]+)?[fh]?)|(-?[0-9]+(e|E)(\+|-)?[0-9]+[fh]?)|(-?[0-9]+[fh])/),hex_float_literal:new F("hex_float_literal",B.token,/-?0x((([0-9a-fA-F]*\.[0-9a-fA-F]+|[0-9a-fA-F]+\.[0-9a-fA-F]*)((p|P)(\+|-)?[0-9]+[fh]?)?)|([0-9a-fA-F]+(p|P)(\+|-)?[0-9]+[fh]?))/),int_literal:new F("int_literal",B.token,/-?0x[0-9a-fA-F]+|0i?|-?[1-9][0-9]*i?/),uint_literal:new F("uint_literal",B.token,/0x[0-9a-fA-F]+u|0u|[1-9][0-9]*u/),name:new F("name",B.token,/([_\p{XID_Start}][\p{XID_Continue}]+)|([\p{XID_Start}])/u),ident:new F("ident",B.token,/[_a-zA-Z][0-9a-zA-Z_]*/),and:new F("and",B.token,"&"),and_and:new F("and_and",B.token,"&&"),arrow:new F("arrow ",B.token,"->"),attr:new F("attr",B.token,"@"),forward_slash:new F("forward_slash",B.token,"/"),bang:new F("bang",B.token,"!"),bracket_left:new F("bracket_left",B.token,"["),bracket_right:new F("bracket_right",B.token,"]"),brace_left:new F("brace_left",B.token,"{"),brace_right:new F("brace_right",B.token,"}"),colon:new F("colon",B.token,":"),comma:new F("comma",B.token,","),equal:new F("equal",B.token,"="),equal_equal:new F("equal_equal",B.token,"=="),not_equal:new F("not_equal",B.token,"!="),greater_than:new F("greater_than",B.token,">"),greater_than_equal:new F("greater_than_equal",B.token,">="),shift_right:new F("shift_right",B.token,">>"),less_than:new F("less_than",B.token,"<"),less_than_equal:new F("less_than_equal",B.token,"<="),shift_left:new F("shift_left",B.token,"<<"),modulo:new F("modulo",B.token,"%"),minus:new F("minus",B.token,"-"),minus_minus:new F("minus_minus",B.token,"--"),period:new F("period",B.token,"."),plus:new F("plus",B.token,"+"),plus_plus:new F("plus_plus",B.token,"++"),or:new F("or",B.token,"|"),or_or:new F("or_or",B.token,"||"),paren_left:new F("paren_left",B.token,"("),paren_right:new F("paren_right",B.token,")"),semicolon:new F("semicolon",B.token,";"),star:new F("star",B.token,"*"),tilde:new F("tilde",B.token,"~"),underscore:new F("underscore",B.token,"_"),xor:new F("xor",B.token,"^"),plus_equal:new F("plus_equal",B.token,"+="),minus_equal:new F("minus_equal",B.token,"-="),times_equal:new F("times_equal",B.token,"*="),division_equal:new F("division_equal",B.token,"/="),modulo_equal:new F("modulo_equal",B.token,"%="),and_equal:new F("and_equal",B.token,"&="),or_equal:new F("or_equal",B.token,"|="),xor_equal:new F("xor_equal",B.token,"^="),shift_right_equal:new F("shift_right_equal",B.token,">>="),shift_left_equal:new F("shift_left_equal",B.token,"<<=")},D.simpleTokens={"@":V.tokens.attr,"{":V.tokens.brace_left,"}":V.tokens.brace_right,":":V.tokens.colon,",":V.tokens.comma,"(":V.tokens.paren_left,")":V.tokens.paren_right,";":V.tokens.semicolon},D.literalTokens={"&":V.tokens.and,"&&":V.tokens.and_and,"->":V.tokens.arrow,"/":V.tokens.forward_slash,"!":V.tokens.bang,"[":V.tokens.bracket_left,"]":V.tokens.bracket_right,"=":V.tokens.equal,"==":V.tokens.equal_equal,"!=":V.tokens.not_equal,">":V.tokens.greater_than,">=":V.tokens.greater_than_equal,">>":V.tokens.shift_right,"<":V.tokens.less_than,"<=":V.tokens.less_than_equal,"<<":V.tokens.shift_left,"%":V.tokens.modulo,"-":V.tokens.minus,"--":V.tokens.minus_minus,".":V.tokens.period,"+":V.tokens.plus,"++":V.tokens.plus_plus,"|":V.tokens.or,"||":V.tokens.or_or,"*":V.tokens.star,"~":V.tokens.tilde,_:V.tokens.underscore,"^":V.tokens.xor,"+=":V.tokens.plus_equal,"-=":V.tokens.minus_equal,"*=":V.tokens.times_equal,"/=":V.tokens.division_equal,"%=":V.tokens.modulo_equal,"&=":V.tokens.and_equal,"|=":V.tokens.or_equal,"^=":V.tokens.xor_equal,">>=":V.tokens.shift_right_equal,"<<=":V.tokens.shift_left_equal},D.regexTokens={decimal_float_literal:V.tokens.decimal_float_literal,hex_float_literal:V.tokens.hex_float_literal,int_literal:V.tokens.int_literal,uint_literal:V.tokens.uint_literal,ident:V.tokens.ident},D.storage_class=[V.keywords.function,V.keywords.private,V.keywords.workgroup,V.keywords.uniform,V.keywords.storage],D.access_mode=[V.keywords.read,V.keywords.write,V.keywords.read_write],D.sampler_type=[V.keywords.sampler,V.keywords.sampler_comparison],D.sampled_texture_type=[V.keywords.texture_1d,V.keywords.texture_2d,V.keywords.texture_2d_array,V.keywords.texture_3d,V.keywords.texture_cube,V.keywords.texture_cube_array],D.multisampled_texture_type=[V.keywords.texture_multisampled_2d],D.storage_texture_type=[V.keywords.texture_storage_1d,V.keywords.texture_storage_2d,V.keywords.texture_storage_2d_array,V.keywords.texture_storage_3d],D.depth_texture_type=[V.keywords.texture_depth_2d,V.keywords.texture_depth_2d_array,V.keywords.texture_depth_cube,V.keywords.texture_depth_cube_array,V.keywords.texture_depth_multisampled_2d],D.texture_external_type=[V.keywords.texture_external],D.any_texture_type=[...V.sampled_texture_type,...V.multisampled_texture_type,...V.storage_texture_type,...V.depth_texture_type,...V.texture_external_type],D.texel_format=[V.keywords.r8unorm,V.keywords.r8snorm,V.keywords.r8uint,V.keywords.r8sint,V.keywords.r16uint,V.keywords.r16sint,V.keywords.r16float,V.keywords.rg8unorm,V.keywords.rg8snorm,V.keywords.rg8uint,V.keywords.rg8sint,V.keywords.r32uint,V.keywords.r32sint,V.keywords.r32float,V.keywords.rg16uint,V.keywords.rg16sint,V.keywords.rg16float,V.keywords.rgba8unorm,V.keywords.rgba8unorm_srgb,V.keywords.rgba8snorm,V.keywords.rgba8uint,V.keywords.rgba8sint,V.keywords.bgra8unorm,V.keywords.bgra8unorm_srgb,V.keywords.rgb10a2unorm,V.keywords.rg11b10float,V.keywords.rg32uint,V.keywords.rg32sint,V.keywords.rg32float,V.keywords.rgba16uint,V.keywords.rgba16sint,V.keywords.rgba16float,V.keywords.rgba32uint,V.keywords.rgba32sint,V.keywords.rgba32float],D.const_literal=[V.tokens.int_literal,V.tokens.uint_literal,V.tokens.decimal_float_literal,V.tokens.hex_float_literal,V.keywords.true,V.keywords.false],D.literal_or_ident=[V.tokens.ident,V.tokens.int_literal,V.tokens.uint_literal,V.tokens.decimal_float_literal,V.tokens.hex_float_literal,V.tokens.name],D.element_count_expression=[V.tokens.int_literal,V.tokens.uint_literal,V.tokens.ident],D.template_types=[V.keywords.vec2,V.keywords.vec3,V.keywords.vec4,V.keywords.mat2x2,V.keywords.mat2x3,V.keywords.mat2x4,V.keywords.mat3x2,V.keywords.mat3x3,V.keywords.mat3x4,V.keywords.mat4x2,V.keywords.mat4x3,V.keywords.mat4x4,V.keywords.atomic,V.keywords.bitcast,...V.any_texture_type],D.attribute_name=[V.tokens.ident,V.keywords.block,V.keywords.diagnostic],D.assignment_operators=[V.tokens.equal,V.tokens.plus_equal,V.tokens.minus_equal,V.tokens.times_equal,V.tokens.division_equal,V.tokens.modulo_equal,V.tokens.and_equal,V.tokens.or_equal,V.tokens.xor_equal,V.tokens.shift_right_equal,V.tokens.shift_left_equal],D.increment_operators=[V.tokens.plus_plus,V.tokens.minus_minus];class Bc{constructor(e,t,n,r,i){this.type=e,this.lexeme=t,this.line=n,this.start=r,this.end=i}toString(){return this.lexeme}isTemplateType(){return D.template_types.indexOf(this.type)!=-1}isArrayType(){return this.type==D.keywords.array}isArrayOrTemplateType(){return this.isArrayType()||this.isTemplateType()}}class Gk{constructor(e){this._tokens=[],this._start=0,this._current=0,this._line=1,this._source=e??""}scanTokens(){for(;!this._isAtEnd();)if(this._start=this._current,!this.scanToken())throw`Invalid syntax at line ${this._line}`;return this._tokens.push(new Bc(D.eof,"",this._line,this._current,this._current)),this._tokens}scanToken(){let e=this._advance();if(e==`
`)return this._line++,!0;if(this._isWhitespace(e))return!0;if(e=="/"){if(this._peekAhead()=="/"){for(;e!=`
`;){if(this._isAtEnd())return!0;e=this._advance()}return this._line++,!0}if(this._peekAhead()=="*"){this._advance();let o=1;for(;o>0;){if(this._isAtEnd())return!0;if(e=this._advance(),e==`
`)this._line++;else if(e=="*"){if(this._peekAhead()=="/"&&(this._advance(),o--,o==0))return!0}else e=="/"&&this._peekAhead()=="*"&&(this._advance(),o++)}return!0}}const t=D.simpleTokens[e];if(t)return this._addToken(t),!0;let n=D.none;const r=this._isAlpha(e),i=e==="_";if(this._isAlphaNumeric(e)){let o=this._peekAhead();for(;this._isAlphaNumeric(o);)e+=this._advance(),o=this._peekAhead()}if(r){const o=D.keywords[e];if(o)return this._addToken(o),!0}if(r||i)return this._addToken(D.tokens.ident),!0;for(;;){let o=this._findType(e);const a=this._peekAhead();if(e=="-"&&this._tokens.length>0){if(a=="=")return this._current++,e+=a,this._addToken(D.tokens.minus_equal),!0;if(a=="-")return this._current++,e+=a,this._addToken(D.tokens.minus_minus),!0;const l=this._tokens.length-1;if((D.literal_or_ident.indexOf(this._tokens[l].type)!=-1||this._tokens[l].type==D.tokens.paren_right)&&a!=">")return this._addToken(o),!0}if(e==">"&&(a==">"||a=="=")){let l=!1,u=this._tokens.length-1;for(let c=0;c<5&&u>=0&&D.assignment_operators.indexOf(this._tokens[u].type)===-1;++c,--u)if(this._tokens[u].type===D.tokens.less_than){u>0&&this._tokens[u-1].isArrayOrTemplateType()&&(l=!0);break}if(l)return this._addToken(o),!0}if(o===D.none){let l=e,u=0;const c=2;for(let h=0;h<c;++h)if(l+=this._peekAhead(h),o=this._findType(l),o!==D.none){u=h;break}if(o===D.none)return n!==D.none&&(this._current--,this._addToken(n),!0);e=l,this._current+=u+1}if(n=o,this._isAtEnd())break;e+=this._advance()}return n!==D.none&&(this._addToken(n),!0)}_findType(e){for(const t in D.regexTokens){const n=D.regexTokens[t];if(this._match(e,n.rule))return n}return D.literalTokens[e]||D.none}_match(e,t){const n=t.exec(e);return n&&n.index==0&&n[0]==e}_isAtEnd(){return this._current>=this._source.length}_isAlpha(e){return!this._isNumeric(e)&&!this._isWhitespace(e)&&e!=="_"&&e!=="."&&e!=="("&&e!==")"&&e!=="["&&e!=="]"&&e!=="{"&&e!=="}"&&e!==","&&e!==";"&&e!==":"&&e!=="="&&e!=="!"&&e!=="<"&&e!==">"&&e!=="+"&&e!=="-"&&e!=="*"&&e!=="/"&&e!=="%"&&e!=="&"&&e!=="|"&&e!=="^"&&e!=="~"&&e!=="@"&&e!=="#"&&e!=="?"&&e!=="'"&&e!=="`"&&e!=='"'&&e!=="\\"&&e!==`
`&&e!=="\r"&&e!=="	"&&e!=="\0"}_isNumeric(e){return e>="0"&&e<="9"}_isAlphaNumeric(e){return this._isAlpha(e)||this._isNumeric(e)||e==="_"}_isWhitespace(e){return e==" "||e=="	"||e=="\r"}_advance(e=0){let t=this._source[this._current];return e=e||0,e++,this._current+=e,t}_peekAhead(e=0){return e=e||0,this._current+e>=this._source.length?"\0":this._source[this._current+e]}_addToken(e){const t=this._source.substring(this._start,this._current);this._tokens.push(new Bc(e,t,this._line,this._start,this._current))}}function ee(s){return Array.isArray(s)||s?.buffer instanceof ArrayBuffer}const oo=new Float32Array(1),Wk=new Uint32Array(oo.buffer),qk=new Uint32Array(oo.buffer),ao=new Int32Array(1),Hk=new Float32Array(ao.buffer),jk=new Uint32Array(ao.buffer),lo=new Uint32Array(1),Kk=new Float32Array(lo.buffer),Xk=new Int32Array(lo.buffer);function Fc(s,e,t){if(e===t)return s;if(e==="f32"){if(t==="i32"||t==="x32")return oo[0]=s,Wk[0];if(t==="u32")return oo[0]=s,qk[0]}else if(e==="i32"||e==="x32"){if(t==="f32")return ao[0]=s,Hk[0];if(t==="u32")return ao[0]=s,jk[0]}else if(e==="u32"){if(t==="f32")return lo[0]=s,Kk[0];if(t==="i32"||t==="x32")return lo[0]=s,Xk[0]}return console.error(`Unsupported cast from ${e} to ${t}`),s}class Yk{constructor(e){this.resources=null,this.inUse=!1,this.info=null,this.node=e}}class ui{constructor(e,t){this.align=e,this.size=t}}class yn{constructor(){this.uniforms=[],this.storage=[],this.textures=[],this.samplers=[],this.aliases=[],this.overrides=[],this.structs=[],this.entry=new Ok,this.functions=[],this._types=new Map,this._functions=new Map}_isStorageTexture(e){return e.name=="texture_storage_1d"||e.name=="texture_storage_2d"||e.name=="texture_storage_2d_array"||e.name=="texture_storage_3d"}updateAST(e){for(const t of e)t instanceof kr&&this._functions.set(t.name,new Yk(t));for(const t of e)if(t instanceof kn){const n=this.getTypeInfo(t,null);n instanceof Ln&&this.structs.push(n)}for(const t of e)if(t instanceof Wl)this.aliases.push(this._getAliasInfo(t));else{if(t instanceof Vl){const n=t,r=this._getAttributeNum(n.attributes,"id",0),i=n.type!=null?this.getTypeInfo(n.type,n.attributes):null;this.overrides.push(new $k(n.name,i,n.attributes,r));continue}if(this._isUniformVar(t)){const n=t,r=this._getAttributeNum(n.attributes,"group",0),i=this._getAttributeNum(n.attributes,"binding",0),o=this.getTypeInfo(n.type,n.attributes),a=new li(n.name,o,r,i,n.attributes,Mn.Uniform,n.access);a.access||(a.access="read"),this.uniforms.push(a);continue}if(this._isStorageVar(t)){const n=t,r=this._getAttributeNum(n.attributes,"group",0),i=this._getAttributeNum(n.attributes,"binding",0),o=this.getTypeInfo(n.type,n.attributes),a=this._isStorageTexture(o),l=new li(n.name,o,r,i,n.attributes,a?Mn.StorageTexture:Mn.Storage,n.access);l.access||(l.access="read"),this.storage.push(l);continue}if(this._isTextureVar(t)){const n=t,r=this._getAttributeNum(n.attributes,"group",0),i=this._getAttributeNum(n.attributes,"binding",0),o=this.getTypeInfo(n.type,n.attributes),a=this._isStorageTexture(o),l=new li(n.name,o,r,i,n.attributes,a?Mn.StorageTexture:Mn.Texture,n.access);l.access||(l.access="read"),a?this.storage.push(l):this.textures.push(l);continue}if(this._isSamplerVar(t)){const n=t,r=this._getAttributeNum(n.attributes,"group",0),i=this._getAttributeNum(n.attributes,"binding",0),o=this.getTypeInfo(n.type,n.attributes),a=new li(n.name,o,r,i,n.attributes,Mn.Sampler,n.access);this.samplers.push(a);continue}}for(const t of e)if(t instanceof kr){const n=this._getAttribute(t,"vertex"),r=this._getAttribute(t,"fragment"),i=this._getAttribute(t,"compute"),o=n||r||i,a=new Dk(t.name,o?.name,t.attributes);a.attributes=t.attributes,a.startLine=t.startLine,a.endLine=t.endLine,this.functions.push(a),this._functions.get(t.name).info=a,o&&(this._functions.get(t.name).inUse=!0,a.inUse=!0,a.resources=this._findResources(t,!!o),a.inputs=this._getInputs(t.args),a.outputs=this._getOutputs(t.returnType),this.entry[o.name].push(a)),a.arguments=t.args.map(l=>new Nk(l.name,this.getTypeInfo(l.type,l.attributes),l.attributes)),a.returnType=t.returnType?this.getTypeInfo(t.returnType,t.attributes):null;continue}for(const t of this._functions.values())t.info&&(t.info.inUse=t.inUse,this._addCalls(t.node,t.info.calls));for(const t of this._functions.values())t.node.search(n=>{var r,i,o;if(n instanceof Qp){if(n.value)if(ee(n.value))for(const a of n.value)for(const l of this.overrides)a===l.name&&((r=t.info)===null||r===void 0||r.overrides.push(l));else for(const a of this.overrides)n.value===a.name&&((i=t.info)===null||i===void 0||i.overrides.push(a))}else if(n instanceof Mt)for(const a of this.overrides)n.name===a.name&&((o=t.info)===null||o===void 0||o.overrides.push(a))});for(const t of this.uniforms)this._markStructsInUse(t.type);for(const t of this.storage)this._markStructsInUse(t.type)}getFunctionInfo(e){for(const t of this.functions)if(t.name==e)return t;return null}getStructInfo(e){for(const t of this.structs)if(t.name==e)return t;return null}getOverrideInfo(e){for(const t of this.overrides)if(t.name==e)return t;return null}_markStructsInUse(e){if(e)if(e.isStruct){if(e.inUse=!0,e.members)for(const t of e.members)this._markStructsInUse(t.type)}else if(e.isArray)this._markStructsInUse(e.format);else if(e.isTemplate)e.format&&this._markStructsInUse(e.format);else{const t=this._getAlias(e.name);t&&this._markStructsInUse(t)}}_addCalls(e,t){var n;for(const r of e.calls){const i=(n=this._functions.get(r.name))===null||n===void 0?void 0:n.info;i&&t.add(i)}}findResource(e,t,n){if(n){for(const r of this.entry.compute)if(r.name===n){for(const i of r.resources)if(i.group==e&&i.binding==t)return i}for(const r of this.entry.vertex)if(r.name===n){for(const i of r.resources)if(i.group==e&&i.binding==t)return i}for(const r of this.entry.fragment)if(r.name===n){for(const i of r.resources)if(i.group==e&&i.binding==t)return i}}for(const r of this.uniforms)if(r.group==e&&r.binding==t)return r;for(const r of this.storage)if(r.group==e&&r.binding==t)return r;for(const r of this.textures)if(r.group==e&&r.binding==t)return r;for(const r of this.samplers)if(r.group==e&&r.binding==t)return r;return null}_findResource(e){for(const t of this.uniforms)if(t.name==e)return t;for(const t of this.storage)if(t.name==e)return t;for(const t of this.textures)if(t.name==e)return t;for(const t of this.samplers)if(t.name==e)return t;return null}_markStructsFromAST(e){const t=this.getTypeInfo(e,null);this._markStructsInUse(t)}_findResources(e,t){const n=[],r=this,i=[];return e.search(o=>{if(o instanceof ro)i.push({});else if(o instanceof io)i.pop();else if(o instanceof Cn){const a=o;t&&a.type!==null&&this._markStructsFromAST(a.type),i.length>0&&(i[i.length-1][a.name]=a)}else if(o instanceof fn){const a=o;t&&a.type!==null&&this._markStructsFromAST(a.type)}else if(o instanceof mr){const a=o;t&&a.type!==null&&this._markStructsFromAST(a.type),i.length>0&&(i[i.length-1][a.name]=a)}else if(o instanceof Mt){const a=o;if(i.length>0&&i[i.length-1][a.name])return;const l=r._findResource(a.name);l&&n.push(l)}else if(o instanceof ql){const a=o,l=r._functions.get(a.name);l&&(t&&(l.inUse=!0),e.calls.add(l.node),l.resources===null&&(l.resources=r._findResources(l.node,t)),n.push(...l.resources))}else if(o instanceof Gl){const a=o,l=r._functions.get(a.name);l&&(t&&(l.inUse=!0),e.calls.add(l.node),l.resources===null&&(l.resources=r._findResources(l.node,t)),n.push(...l.resources))}}),[...new Map(n.map(o=>[o.name,o])).values()]}getBindGroups(){const e=[];function t(n,r){n>=e.length&&(e.length=n+1),e[n]===void 0&&(e[n]=[]),r>=e[n].length&&(e[n].length=r+1)}for(const n of this.uniforms)t(n.group,n.binding),e[n.group][n.binding]=n;for(const n of this.storage)t(n.group,n.binding),e[n.group][n.binding]=n;for(const n of this.textures)t(n.group,n.binding),e[n.group][n.binding]=n;for(const n of this.samplers)t(n.group,n.binding),e[n.group][n.binding]=n;return e}_getOutputs(e,t=void 0){if(t===void 0&&(t=[]),e instanceof kn)this._getStructOutputs(e,t);else{const n=this._getOutputInfo(e);n!==null&&t.push(n)}return t}_getStructOutputs(e,t){for(const n of e.members)if(n.type instanceof kn)this._getStructOutputs(n.type,t);else{const r=this._getAttribute(n,"location")||this._getAttribute(n,"builtin");if(r!==null){const i=this.getTypeInfo(n.type,n.type.attributes),o=this._parseInt(r.value),a=new Oc(n.name,i,r.name,o);t.push(a)}}}_getOutputInfo(e){const t=this._getAttribute(e,"location")||this._getAttribute(e,"builtin");if(t!==null){const n=this.getTypeInfo(e,e.attributes),r=this._parseInt(t.value);return new Oc("",n,t.name,r)}return null}_getInputs(e,t=void 0){t===void 0&&(t=[]);for(const n of e)if(n.type instanceof kn)this._getStructInputs(n.type,t);else{const r=this._getInputInfo(n);r!==null&&t.push(r)}return t}_getStructInputs(e,t){for(const n of e.members)if(n.type instanceof kn)this._getStructInputs(n.type,t);else{const r=this._getInputInfo(n);r!==null&&t.push(r)}}_getInputInfo(e){const t=this._getAttribute(e,"location")||this._getAttribute(e,"builtin");if(t!==null){const n=this._getAttribute(e,"interpolation"),r=this.getTypeInfo(e.type,e.attributes),i=this._parseInt(t.value),o=new Ck(e.name,r,t.name,i);return n!==null&&(o.interpolation=this._parseString(n.value)),o}return null}_parseString(e){return e instanceof Array&&(e=e[0]),e}_parseInt(e){e instanceof Array&&(e=e[0]);const t=parseInt(e);return isNaN(t)?e:t}_getAlias(e){for(const t of this.aliases)if(t.name==e)return t.type;return null}_getAliasInfo(e){return new Ak(e.name,this.getTypeInfo(e.type,null))}getTypeInfoByName(e){for(const t of this.structs)if(t.name==e)return t;for(const t of this.aliases)if(t.name==e)return t.type;return null}getTypeInfo(e,t=null){if(this._types.has(e))return this._types.get(e);if(e instanceof Ii){const r=e.type?this.getTypeInfo(e.type,e.attributes):null,i=new Ra(e.name,r,t);return this._types.set(e,i),this._updateTypeInfo(i),i}if(e instanceof gr){const r=e,i=r.format?this.getTypeInfo(r.format,r.attributes):null,o=new Vn(r.name,t);return o.format=i,o.count=r.count,this._types.set(e,o),this._updateTypeInfo(o),o}if(e instanceof kn){const r=e,i=new Ln(r.name,t);i.startLine=r.startLine,i.endLine=r.endLine;for(const o of r.members){const a=this.getTypeInfo(o.type,o.attributes);i.members.push(new Dc(o.name,a,o.attributes))}return this._types.set(e,i),this._updateTypeInfo(i),i}if(e instanceof ur){const r=e,i=r.format instanceof K,o=r.format?i?this.getTypeInfo(r.format,null):new Gt(r.format,null):null,a=new ps(r.name,o,t,r.access);return this._types.set(e,a),this._updateTypeInfo(a),a}if(e instanceof U){const r=e,i=r.format?this.getTypeInfo(r.format,null):null,o=new ps(r.name,i,t,r.access);return this._types.set(e,o),this._updateTypeInfo(o),o}const n=new Gt(e.name,t);return this._types.set(e,n),this._updateTypeInfo(n),n}_updateTypeInfo(e){var t,n,r;const i=this._getTypeSize(e);if(e.size=(t=i?.size)!==null&&t!==void 0?t:0,e instanceof Vn&&e.format){const o=this._getTypeSize(e.format);e.stride=Math.max((n=o?.size)!==null&&n!==void 0?n:0,(r=o?.align)!==null&&r!==void 0?r:0),this._updateTypeInfo(e.format)}e instanceof Ra&&this._updateTypeInfo(e.format),e instanceof Ln&&this._updateStructInfo(e)}_updateStructInfo(e){var t;let n=0,r=0,i=0,o=0;for(let a=0,l=e.members.length;a<l;++a){const u=e.members[a],c=this._getTypeSize(u);if(!c)continue;(t=this._getAlias(u.type.name))!==null&&t!==void 0||u.type;const h=c.align,d=c.size;n=this._roundUp(h,n+r),r=d,i=n,o=Math.max(o,h),u.offset=n,u.size=d,this._updateTypeInfo(u.type)}e.size=this._roundUp(o,i+r),e.align=o}_getTypeSize(e){var t,n;if(e==null)return null;const r=this._getAttributeNum(e.attributes,"size",0),i=this._getAttributeNum(e.attributes,"align",0);if(e instanceof Dc&&(e=e.type),e instanceof Gt){const o=this._getAlias(e.name);o!==null&&(e=o)}{const o=yn._typeInfo[e.name];if(o!==void 0){const a=((t=e.format)===null||t===void 0?void 0:t.name)==="f16"?2:1;return new ui(Math.max(i,o.align/a),Math.max(r,o.size/a))}}{const o=yn._typeInfo[e.name.substring(0,e.name.length-1)];if(o){const a=e.name[e.name.length-1]==="h"?2:1;return new ui(Math.max(i,o.align/a),Math.max(r,o.size/a))}}if(e instanceof Vn){let o=e,a=8,l=8;const u=this._getTypeSize(o.format);return u!==null&&(l=u.size,a=u.align),l=o.count*this._getAttributeNum((n=e?.attributes)!==null&&n!==void 0?n:null,"stride",this._roundUp(a,l)),r&&(l=r),new ui(Math.max(i,a),Math.max(r,l))}if(e instanceof Ln){let o=0,a=0,l=0,u=0,c=0;for(const h of e.members){const d=this._getTypeSize(h.type);d!==null&&(o=Math.max(d.align,o),l=this._roundUp(d.align,l+u),u=d.size,c=l)}return a=this._roundUp(o,c+u),new ui(Math.max(i,o),Math.max(r,a))}return null}_isUniformVar(e){return e instanceof Cn&&e.storage=="uniform"}_isStorageVar(e){return e instanceof Cn&&e.storage=="storage"}_isTextureVar(e){return e instanceof Cn&&e.type!==null&&yn._textureTypes.indexOf(e.type.name)!=-1}_isSamplerVar(e){return e instanceof Cn&&e.type!==null&&yn._samplerTypes.indexOf(e.type.name)!=-1}_getAttribute(e,t){const n=e;if(!n||!n.attributes)return null;const r=n.attributes;for(let i of r)if(i.name==t)return i;return null}_getAttributeNum(e,t,n){if(e===null)return n;for(let r of e)if(r.name==t){let i=r!==null&&r.value!==null?r.value:n;return i instanceof Array&&(i=i[0]),typeof i=="number"?i:typeof i=="string"?parseInt(i):n}return n}_roundUp(e,t){return Math.ceil(t/e)*e}}yn._typeInfo={f16:{align:2,size:2},i32:{align:4,size:4},u32:{align:4,size:4},f32:{align:4,size:4},atomic:{align:4,size:4},vec2:{align:8,size:8},vec3:{align:16,size:12},vec4:{align:16,size:16},mat2x2:{align:8,size:16},mat3x2:{align:8,size:24},mat4x2:{align:8,size:32},mat2x3:{align:16,size:32},mat3x3:{align:16,size:48},mat4x3:{align:16,size:64},mat2x4:{align:16,size:32},mat3x4:{align:16,size:48},mat4x4:{align:16,size:64}},yn._textureTypes=D.any_texture_type.map(s=>s.name),yn._samplerTypes=D.sampler_type.map(s=>s.name);let Hl=0;class jl{constructor(e,t,n){this.id=Hl++,this.name=e,this.value=t,this.node=n}clone(){return new jl(this.name,this.value,this.node)}}class Kl{constructor(e){this.id=Hl++,this.name=e.name,this.node=e}clone(){return new Kl(this.node)}}class Xl{constructor(e){this.parent=null,this.variables=new Map,this.functions=new Map,this.currentFunctionName="",this.id=Hl++,e&&(this.parent=e,this.currentFunctionName=e.currentFunctionName)}getVariable(e){var t;return this.variables.has(e)?(t=this.variables.get(e))!==null&&t!==void 0?t:null:this.parent?this.parent.getVariable(e):null}getFunction(e){var t;return this.functions.has(e)?(t=this.functions.get(e))!==null&&t!==void 0?t:null:this.parent?this.parent.getFunction(e):null}createVariable(e,t,n){this.variables.set(e,new jl(e,t,n??null))}setVariable(e,t,n){const r=this.getVariable(e);r!==null?r.value=t:this.createVariable(e,t,n)}getVariableValue(e){var t;return(t=this.getVariable(e)?.value)!==null&&t!==void 0?t:null}clone(){return new Xl(this)}}class Qk{evalExpression(e,t){return null}getTypeInfo(e){return null}getVariableName(e,t){return""}}class Zk{constructor(e){this.exec=e}getTypeInfo(e){return this.exec.getTypeInfo(e)}All(e,t){const n=this.exec.evalExpression(e.args[0],t);let r=!0;if(n instanceof M)return n.data.forEach(i=>{i||(r=!1)}),new L(r?1:0,this.getTypeInfo("bool"));throw new Error(`All() expects a vector argument. Line ${e.line}`)}Any(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M){const r=n.data.some(i=>i);return new L(r?1:0,this.getTypeInfo("bool"))}throw new Error(`Any() expects a vector argument. Line ${e.line}`)}Select(e,t){const n=this.exec.evalExpression(e.args[2],t);if(!(n instanceof L))throw new Error(`Select() expects a bool condition. Line ${e.line}`);return n.value?this.exec.evalExpression(e.args[1],t):this.exec.evalExpression(e.args[0],t)}ArrayLength(e,t){let n=e.args[0];n instanceof Ue&&(n=n.right);const r=this.exec.evalExpression(n,t);if(r instanceof Le&&r.typeInfo.size===0){const i=r.typeInfo,o=r.buffer.byteLength/i.stride;return new L(o,this.getTypeInfo("u32"))}return new L(r.typeInfo.size,this.getTypeInfo("u32"))}Abs(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.abs(i)),n.typeInfo);const r=n;return new L(Math.abs(r.value),r.typeInfo)}Acos(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.acos(i)),n.typeInfo);const r=n;return new L(Math.acos(r.value),n.typeInfo)}Acosh(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.acosh(i)),n.typeInfo);const r=n;return new L(Math.acosh(r.value),n.typeInfo)}Asin(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.asin(i)),n.typeInfo);const r=n;return new L(Math.asin(r.value),n.typeInfo)}Asinh(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.asinh(i)),n.typeInfo);const r=n;return new L(Math.asinh(r.value),n.typeInfo)}Atan(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.atan(i)),n.typeInfo);const r=n;return new L(Math.atan(r.value),n.typeInfo)}Atanh(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.atanh(i)),n.typeInfo);const r=n;return new L(Math.atanh(r.value),n.typeInfo)}Atan2(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);if(n instanceof M&&r instanceof M)return new M(n.data.map((a,l)=>Math.atan2(a,r.data[l])),n.typeInfo);const i=n,o=r;return new L(Math.atan2(i.value,o.value),n.typeInfo)}Ceil(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.ceil(i)),n.typeInfo);const r=n;return new L(Math.ceil(r.value),n.typeInfo)}_clamp(e,t,n){return Math.min(Math.max(e,t),n)}Clamp(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t),i=this.exec.evalExpression(e.args[2],t);if(n instanceof M&&r instanceof M&&i instanceof M)return new M(n.data.map((u,c)=>this._clamp(u,r.data[c],i.data[c])),n.typeInfo);const o=n,a=r,l=i;return new L(this._clamp(o.value,a.value,l.value),n.typeInfo)}Cos(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.cos(i)),n.typeInfo);const r=n;return new L(Math.cos(r.value),n.typeInfo)}Cosh(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.cosh(i)),n.typeInfo);const r=n;return new L(Math.cos(r.value),n.typeInfo)}CountLeadingZeros(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.clz32(i)),n.typeInfo);const r=n;return new L(Math.clz32(r.value),n.typeInfo)}_countOneBits(e){let t=0;for(;e!==0;)1&e&&t++,e>>=1;return t}CountOneBits(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>this._countOneBits(i)),n.typeInfo);const r=n;return new L(this._countOneBits(r.value),n.typeInfo)}_countTrailingZeros(e){if(e===0)return 32;let t=0;for(;!(1&e);)e>>=1,t++;return t}CountTrailingZeros(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>this._countTrailingZeros(i)),n.typeInfo);const r=n;return new L(this._countTrailingZeros(r.value),n.typeInfo)}Cross(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);if(n instanceof M&&r instanceof M){if(n.data.length!==3||r.data.length!==3)return console.error(`Cross() expects 3D vectors. Line ${e.line}`),null;const i=n.data,o=r.data;return new M([i[1]*o[2]-o[1]*i[2],i[2]*o[0]-o[2]*i[0],i[0]*o[1]-o[0]*i[1]],n.typeInfo)}return console.error(`Cross() expects vector arguments. Line ${e.line}`),null}Degrees(e,t){const n=this.exec.evalExpression(e.args[0],t),r=180/Math.PI;return n instanceof M?new M(n.data.map(i=>i*r),n.typeInfo):new L(n.value*r,this.getTypeInfo("f32"))}Determinant(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof ue){const r=n.data,i=n.typeInfo.getTypeName(),o=i.endsWith("h")?this.getTypeInfo("f16"):this.getTypeInfo("f32");if(i==="mat2x2"||i==="mat2x2f"||i==="mat2x2h")return new L(r[0]*r[3]-r[1]*r[2],o);if(i==="mat2x3"||i==="mat2x3f"||i==="mat2x3h")return new L(r[0]*(r[4]*r[8]-r[5]*r[7])-r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6]),o);if(i==="mat2x4"||i==="mat2x4f"||i==="mat2x4h")console.error(`TODO: Determinant for ${i}`);else if(i==="mat3x2"||i==="mat3x2f"||i==="mat3x2h")console.error(`TODO: Determinant for ${i}`);else{if(i==="mat3x3"||i==="mat3x3f"||i==="mat3x3h")return new L(r[0]*(r[4]*r[8]-r[5]*r[7])-r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6]),o);i==="mat3x4"||i==="mat3x4f"||i==="mat3x4h"||i==="mat4x2"||i==="mat4x2f"||i==="mat4x2h"||i==="mat4x3"||i==="mat4x3f"||i==="mat4x3h"?console.error(`TODO: Determinant for ${i}`):i!=="mat4x4"&&i!=="mat4x4f"&&i!=="mat4x4h"||console.error(`TODO: Determinant for ${i}`)}}return console.error(`Determinant expects a matrix argument. Line ${e.line}`),null}Distance(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);if(n instanceof M&&r instanceof M){let a=0;for(let l=0;l<n.data.length;++l)a+=(n.data[l]-r.data[l])*(n.data[l]-r.data[l]);return new L(Math.sqrt(a),this.getTypeInfo("f32"))}const i=n,o=r;return new L(Math.abs(i.value-o.value),n.typeInfo)}_dot(e,t){let n=0;for(let r=0;r<e.length;++r)n+=t[r]*e[r];return n}Dot(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);return n instanceof M&&r instanceof M?new L(this._dot(n.data,r.data),this.getTypeInfo("f32")):(console.error(`Dot() expects vector arguments. Line ${e.line}`),null)}Dot4U8Packed(e,t){return console.error(`TODO: dot4U8Packed. Line ${e.line}`),null}Dot4I8Packed(e,t){return console.error(`TODO: dot4I8Packed. Line ${e.line}`),null}Exp(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.exp(i)),n.typeInfo);const r=n;return new L(Math.exp(r.value),n.typeInfo)}Exp2(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.pow(2,i)),n.typeInfo);const r=n;return new L(Math.pow(2,r.value),n.typeInfo)}ExtractBits(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t),i=this.exec.evalExpression(e.args[2],t);if(r.typeInfo.name!=="u32"&&r.typeInfo.name!=="x32")return console.error(`ExtractBits() expects an i32 offset argument. Line ${e.line}`),null;if(i.typeInfo.name!=="u32"&&i.typeInfo.name!=="x32")return console.error(`ExtractBits() expects an i32 count argument. Line ${e.line}`),null;const o=r.value,a=i.value;if(n instanceof M)return new M(n.data.map(u=>u>>o&(1<<a)-1),n.typeInfo);if(n.typeInfo.name!=="i32"&&n.typeInfo.name!=="x32")return console.error(`ExtractBits() expects an i32 argument. Line ${e.line}`),null;const l=n.value;return new L(l>>o&(1<<a)-1,this.getTypeInfo("i32"))}FaceForward(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t),i=this.exec.evalExpression(e.args[2],t);if(n instanceof M&&r instanceof M&&i instanceof M){const o=this._dot(r.data,i.data);return new M(o<0?Array.from(n.data):n.data.map(a=>-a),n.typeInfo)}return console.error(`FaceForward() expects vector arguments. Line ${e.line}`),null}_firstLeadingBit(e){return e===0?-1:31-Math.clz32(e)}FirstLeadingBit(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>this._firstLeadingBit(i)),n.typeInfo);const r=n;return new L(this._firstLeadingBit(r.value),n.typeInfo)}_firstTrailingBit(e){return e===0?-1:Math.log2(e&-e)}FirstTrailingBit(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>this._firstTrailingBit(i)),n.typeInfo);const r=n;return new L(this._firstTrailingBit(r.value),n.typeInfo)}Floor(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.floor(i)),n.typeInfo);const r=n;return new L(Math.floor(r.value),n.typeInfo)}Fma(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t),i=this.exec.evalExpression(e.args[2],t);if(n instanceof M&&r instanceof M&&i instanceof M)return n.data.length!==r.data.length||n.data.length!==i.data.length?(console.error(`Fma() expects vectors of the same length. Line ${e.line}`),null):new M(n.data.map((u,c)=>u*r.data[c]+i.data[c]),n.typeInfo);const o=n,a=r,l=i;return new L(o.value*a.value+l.value,o.typeInfo)}Fract(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>i-Math.floor(i)),n.typeInfo);const r=n;return new L(r.value-Math.floor(r.value),n.typeInfo)}Frexp(e,t){return console.error(`TODO: frexp. Line ${e.line}`),null}InsertBits(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t),i=this.exec.evalExpression(e.args[2],t),o=this.exec.evalExpression(e.args[3],t);if(i.typeInfo.name!=="u32"&&i.typeInfo.name!=="x32")return console.error(`InsertBits() expects an i32 offset argument. Line ${e.line}`),null;const a=i.value,l=(1<<o.value)-1<<a,u=~l;if(n instanceof M&&r instanceof M)return new M(n.data.map((d,w)=>d&u|r.data[w]<<a&l),n.typeInfo);const c=n.value,h=r.value;return new L(c&u|h<<a&l,n.typeInfo)}InverseSqrt(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>1/Math.sqrt(i)),n.typeInfo);const r=n;return new L(1/Math.sqrt(r.value),n.typeInfo)}Ldexp(e,t){return console.error(`TODO: ldexp. Line ${e.line}`),null}Length(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M){let i=0;return n.data.forEach(o=>{i+=o*o}),new L(Math.sqrt(i),this.getTypeInfo("f32"))}const r=n;return new L(Math.abs(r.value),n.typeInfo)}Log(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.log(i)),n.typeInfo);const r=n;return new L(Math.log(r.value),n.typeInfo)}Log2(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.log2(i)),n.typeInfo);const r=n;return new L(Math.log2(r.value),n.typeInfo)}Max(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);if(n instanceof M&&r instanceof M)return new M(n.data.map((a,l)=>Math.max(a,r.data[l])),n.typeInfo);const i=n,o=r;return new L(Math.max(i.value,o.value),n.typeInfo)}Min(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);if(n instanceof M&&r instanceof M)return new M(n.data.map((a,l)=>Math.min(a,r.data[l])),n.typeInfo);const i=n,o=r;return new L(Math.min(i.value,o.value),n.typeInfo)}Mix(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t),i=this.exec.evalExpression(e.args[2],t);if(n instanceof M&&r instanceof M&&i instanceof M)return new M(n.data.map((l,u)=>n.data[u]*(1-i.data[u])+r.data[u]*i.data[u]),n.typeInfo);const o=r,a=i;return new L(n.value*(1-a.value)+o.value*a.value,n.typeInfo)}Modf(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);if(n instanceof M&&r instanceof M)return new M(n.data.map((o,a)=>o%r.data[a]),n.typeInfo);const i=r;return new L(n.value%i.value,n.typeInfo)}Normalize(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M){const r=this.Length(e,t).value;return new M(n.data.map(i=>i/r),n.typeInfo)}return console.error(`Normalize() expects a vector argument. Line ${e.line}`),null}Pow(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);if(n instanceof M&&r instanceof M)return new M(n.data.map((a,l)=>Math.pow(a,r.data[l])),n.typeInfo);const i=n,o=r;return new L(Math.pow(i.value,o.value),n.typeInfo)}QuantizeToF16(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof M?new M(n.data.map(r=>r),n.typeInfo):new L(n.value,n.typeInfo)}Radians(e,t){const n=this.exec.evalExpression(e.args[0],t);return n instanceof M?new M(n.data.map(r=>r*Math.PI/180),n.typeInfo):new L(n.value*Math.PI/180,this.getTypeInfo("f32"))}Reflect(e,t){let n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);if(n instanceof M&&r instanceof M){const i=this._dot(n.data,r.data);return new M(n.data.map((o,a)=>o-2*i*r.data[a]),n.typeInfo)}return console.error(`Reflect() expects vector arguments. Line ${e.line}`),null}Refract(e,t){let n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t),i=this.exec.evalExpression(e.args[2],t);if(n instanceof M&&r instanceof M&&i instanceof L){const o=this._dot(r.data,n.data);return new M(n.data.map((a,l)=>{const u=1-i.value*i.value*(1-o*o);if(u<0)return 0;const c=Math.sqrt(u);return i.value*a-(i.value*o+c)*r.data[l]}),n.typeInfo)}return console.error(`Refract() expects vector arguments and a scalar argument. Line ${e.line}`),null}ReverseBits(e,t){return console.error(`TODO: reverseBits. Line ${e.line}`),null}Round(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.round(i)),n.typeInfo);const r=n;return new L(Math.round(r.value),n.typeInfo)}Saturate(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.min(Math.max(i,0),1)),n.typeInfo);const r=n;return new L(Math.min(Math.max(r.value,0),1),n.typeInfo)}Sign(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.sign(i)),n.typeInfo);const r=n;return new L(Math.sign(r.value),n.typeInfo)}Sin(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.sin(i)),n.typeInfo);const r=n;return new L(Math.sin(r.value),n.typeInfo)}Sinh(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.sinh(i)),n.typeInfo);const r=n;return new L(Math.sinh(r.value),n.typeInfo)}_smoothstep(e,t,n){const r=Math.min(Math.max((n-e)/(t-e),0),1);return r*r*(3-2*r)}SmoothStep(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t),i=this.exec.evalExpression(e.args[2],t);if(i instanceof M&&n instanceof M&&r instanceof M)return new M(i.data.map((u,c)=>this._smoothstep(n.data[c],r.data[c],u)),i.typeInfo);const o=n,a=r,l=i;return new L(this._smoothstep(o.value,a.value,l.value),i.typeInfo)}Sqrt(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.sqrt(i)),n.typeInfo);const r=n;return new L(Math.sqrt(r.value),n.typeInfo)}Step(e,t){const n=this.exec.evalExpression(e.args[0],t),r=this.exec.evalExpression(e.args[1],t);if(r instanceof M&&n instanceof M)return new M(r.data.map((o,a)=>o<n.data[a]?0:1),r.typeInfo);const i=n;return new L(r.value<i.value?0:1,i.typeInfo)}Tan(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.tan(i)),n.typeInfo);const r=n;return new L(Math.tan(r.value),n.typeInfo)}Tanh(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.tanh(i)),n.typeInfo);const r=n;return new L(Math.tanh(r.value),n.typeInfo)}_getTransposeType(e){const t=e.getTypeName();return t==="mat2x2f"||t==="mat2x2h"?e:t==="mat2x3f"?this.getTypeInfo("mat3x2f"):t==="mat2x3h"?this.getTypeInfo("mat3x2h"):t==="mat2x4f"?this.getTypeInfo("mat4x2f"):t==="mat2x4h"?this.getTypeInfo("mat4x2h"):t==="mat3x2f"?this.getTypeInfo("mat2x3f"):t==="mat3x2h"?this.getTypeInfo("mat2x3h"):t==="mat3x3f"||t==="mat3x3h"?e:t==="mat3x4f"?this.getTypeInfo("mat4x3f"):t==="mat3x4h"?this.getTypeInfo("mat4x3h"):t==="mat4x2f"?this.getTypeInfo("mat2x4f"):t==="mat4x2h"?this.getTypeInfo("mat2x4h"):t==="mat4x3f"?this.getTypeInfo("mat3x4f"):t==="mat4x3h"?this.getTypeInfo("mat3x4h"):(t==="mat4x4f"||t==="mat4x4h"||console.error(`Invalid matrix type ${t}`),e)}Transpose(e,t){const n=this.exec.evalExpression(e.args[0],t);if(!(n instanceof ue))return console.error(`Transpose() expects a matrix argument. Line ${e.line}`),null;const r=this._getTransposeType(n.typeInfo);if(n.typeInfo.name==="mat2x2"||n.typeInfo.name==="mat2x2f"||n.typeInfo.name==="mat2x2h"){const i=n.data;return new ue([i[0],i[2],i[1],i[3]],r)}if(n.typeInfo.name==="mat2x3"||n.typeInfo.name==="mat2x3f"||n.typeInfo.name==="mat2x3h"){const i=n.data;return new ue([i[0],i[3],i[6],i[1],i[4],i[7]],r)}if(n.typeInfo.name==="mat2x4"||n.typeInfo.name==="mat2x4f"||n.typeInfo.name==="mat2x4h"){const i=n.data;return new ue([i[0],i[4],i[8],i[12],i[1],i[5],i[9],i[13]],r)}if(n.typeInfo.name==="mat3x2"||n.typeInfo.name==="mat3x2f"||n.typeInfo.name==="mat3x2h"){const i=n.data;return new ue([i[0],i[3],i[1],i[4],i[2],i[5]],r)}if(n.typeInfo.name==="mat3x3"||n.typeInfo.name==="mat3x3f"||n.typeInfo.name==="mat3x3h"){const i=n.data;return new ue([i[0],i[3],i[6],i[1],i[4],i[7],i[2],i[5],i[8]],r)}if(n.typeInfo.name==="mat3x4"||n.typeInfo.name==="mat3x4f"||n.typeInfo.name==="mat3x4h"){const i=n.data;return new ue([i[0],i[4],i[8],i[12],i[1],i[5],i[9],i[13],i[2],i[6],i[10],i[14]],r)}if(n.typeInfo.name==="mat4x2"||n.typeInfo.name==="mat4x2f"||n.typeInfo.name==="mat4x2h"){const i=n.data;return new ue([i[0],i[4],i[1],i[5],i[2],i[6]],r)}if(n.typeInfo.name==="mat4x3"||n.typeInfo.name==="mat4x3f"||n.typeInfo.name==="mat4x3h"){const i=n.data;return new ue([i[0],i[4],i[8],i[1],i[5],i[9],i[2],i[6],i[10]],r)}if(n.typeInfo.name==="mat4x4"||n.typeInfo.name==="mat4x4f"||n.typeInfo.name==="mat4x4h"){const i=n.data;return new ue([i[0],i[4],i[8],i[12],i[1],i[5],i[9],i[13],i[2],i[6],i[10],i[14],i[3],i[7],i[11],i[15]],r)}return console.error(`Invalid matrix type ${n.typeInfo.name}`),null}Trunc(e,t){const n=this.exec.evalExpression(e.args[0],t);if(n instanceof M)return new M(n.data.map(i=>Math.trunc(i)),n.typeInfo);const r=n;return new L(Math.trunc(r.value),n.typeInfo)}Dpdx(e,t){return console.error(`TODO: dpdx. Line ${e.line}`),null}DpdxCoarse(e,t){return console.error(`TODO: dpdxCoarse. Line ${e.line}`),null}DpdxFine(e,t){return console.error("TODO: dpdxFine"),null}Dpdy(e,t){return console.error("TODO: dpdy"),null}DpdyCoarse(e,t){return console.error("TODO: dpdyCoarse"),null}DpdyFine(e,t){return console.error("TODO: dpdyFine"),null}Fwidth(e,t){return console.error("TODO: fwidth"),null}FwidthCoarse(e,t){return console.error("TODO: fwidthCoarse"),null}FwidthFine(e,t){return console.error("TODO: fwidthFine"),null}TextureDimensions(e,t){const n=e.args[0],r=e.args.length>1?this.exec.evalExpression(e.args[1],t).value:0;if(n instanceof Mt){const i=n.name,o=t.getVariableValue(i);if(o instanceof Tn){if(r<0||r>=o.mipLevelCount)return console.error(`Invalid mip level for textureDimensions. Line ${e.line}`),null;const a=o.getMipLevelSize(r),l=o.dimension;return l==="1d"?new L(a[0],this.getTypeInfo("u32")):l==="3d"?new M(a,this.getTypeInfo("vec3u")):l==="2d"?new M(a.slice(0,2),this.getTypeInfo("vec2u")):(console.error(`Invalid texture dimension ${l} not found. Line ${e.line}`),null)}return console.error(`Texture ${i} not found. Line ${e.line}`),null}return console.error(`Invalid texture argument for textureDimensions. Line ${e.line}`),null}TextureGather(e,t){return console.error("TODO: textureGather"),null}TextureGatherCompare(e,t){return console.error("TODO: textureGatherCompare"),null}TextureLoad(e,t){const n=e.args[0],r=this.exec.evalExpression(e.args[1],t),i=e.args.length>2?this.exec.evalExpression(e.args[2],t).value:0;if(!(r instanceof M)||r.data.length!==2)return console.error(`Invalid UV argument for textureLoad. Line ${e.line}`),null;if(n instanceof Mt){const o=n.name,a=t.getVariableValue(o);if(a instanceof Tn){const l=Math.floor(r.data[0]),u=Math.floor(r.data[1]);if(l<0||l>=a.width||u<0||u>=a.height)return console.error(`Texture ${o} out of bounds. Line ${e.line}`),null;const c=a.getPixel(l,u,0,i);return c===null?(console.error(`Invalid texture format for textureLoad. Line ${e.line}`),null):new M(c,this.getTypeInfo("vec4f"))}return console.error(`Texture ${o} not found. Line ${e.line}`),null}return console.error(`Invalid texture argument for textureLoad. Line ${e.line}`),null}TextureNumLayers(e,t){const n=e.args[0];if(n instanceof Mt){const r=n.name,i=t.getVariableValue(r);return i instanceof Tn?new L(i.depthOrArrayLayers,this.getTypeInfo("u32")):(console.error(`Texture ${r} not found. Line ${e.line}`),null)}return console.error(`Invalid texture argument for textureNumLayers. Line ${e.line}`),null}TextureNumLevels(e,t){const n=e.args[0];if(n instanceof Mt){const r=n.name,i=t.getVariableValue(r);return i instanceof Tn?new L(i.mipLevelCount,this.getTypeInfo("u32")):(console.error(`Texture ${r} not found. Line ${e.line}`),null)}return console.error(`Invalid texture argument for textureNumLevels. Line ${e.line}`),null}TextureNumSamples(e,t){const n=e.args[0];if(n instanceof Mt){const r=n.name,i=t.getVariableValue(r);return i instanceof Tn?new L(i.sampleCount,this.getTypeInfo("u32")):(console.error(`Texture ${r} not found. Line ${e.line}`),null)}return console.error(`Invalid texture argument for textureNumSamples. Line ${e.line}`),null}TextureSample(e,t){return console.error("TODO: textureSample"),null}TextureSampleBias(e,t){return console.error("TODO: textureSampleBias"),null}TextureSampleCompare(e,t){return console.error("TODO: textureSampleCompare"),null}TextureSampleCompareLevel(e,t){return console.error("TODO: textureSampleCompareLevel"),null}TextureSampleGrad(e,t){return console.error("TODO: textureSampleGrad"),null}TextureSampleLevel(e,t){return console.error("TODO: textureSampleLevel"),null}TextureSampleBaseClampToEdge(e,t){return console.error("TODO: textureSampleBaseClampToEdge"),null}TextureStore(e,t){const n=e.args[0],r=this.exec.evalExpression(e.args[1],t),i=e.args.length===4?this.exec.evalExpression(e.args[2],t).value:0,o=e.args.length===4?this.exec.evalExpression(e.args[3],t).data:this.exec.evalExpression(e.args[2],t).data;if(o.length!==4)return console.error(`Invalid value argument for textureStore. Line ${e.line}`),null;if(!(r instanceof M)||r.data.length!==2)return console.error(`Invalid UV argument for textureStore. Line ${e.line}`),null;if(n instanceof Mt){const a=n.name,l=t.getVariableValue(a);if(l instanceof Tn){const u=l.getMipLevelSize(0),c=Math.floor(r.data[0]),h=Math.floor(r.data[1]);return c<0||c>=u[0]||h<0||h>=u[1]?(console.error(`Texture ${a} out of bounds. Line ${e.line}`),null):(l.setPixel(c,h,0,i,Array.from(o)),null)}return console.error(`Texture ${a} not found. Line ${e.line}`),null}return console.error(`Invalid texture argument for textureStore. Line ${e.line}`),null}AtomicLoad(e,t){let n=e.args[0];n instanceof Ue&&(n=n.right);const r=this.exec.getVariableName(n,t);return t.getVariable(r).value.getSubData(this.exec,n.postfix,t)}AtomicStore(e,t){let n=e.args[0];n instanceof Ue&&(n=n.right);const r=this.exec.getVariableName(n,t),i=t.getVariable(r);let o=e.args[1];const a=this.exec.evalExpression(o,t),l=i.value.getSubData(this.exec,n.postfix,t);return l instanceof L&&a instanceof L&&(l.value=a.value),i.value instanceof Le&&i.value.setDataValue(this.exec,l,n.postfix,t),null}AtomicAdd(e,t){let n=e.args[0];n instanceof Ue&&(n=n.right);const r=this.exec.getVariableName(n,t),i=t.getVariable(r);let o=e.args[1];const a=this.exec.evalExpression(o,t),l=i.value.getSubData(this.exec,n.postfix,t),u=new L(l.value,l.typeInfo);return l instanceof L&&a instanceof L&&(l.value+=a.value),i.value instanceof Le&&i.value.setDataValue(this.exec,l,n.postfix,t),u}AtomicSub(e,t){let n=e.args[0];n instanceof Ue&&(n=n.right);const r=this.exec.getVariableName(n,t),i=t.getVariable(r);let o=e.args[1];const a=this.exec.evalExpression(o,t),l=i.value.getSubData(this.exec,n.postfix,t),u=new L(l.value,l.typeInfo);return l instanceof L&&a instanceof L&&(l.value-=a.value),i.value instanceof Le&&i.value.setDataValue(this.exec,l,n.postfix,t),u}AtomicMax(e,t){let n=e.args[0];n instanceof Ue&&(n=n.right);const r=this.exec.getVariableName(n,t),i=t.getVariable(r);let o=e.args[1];const a=this.exec.evalExpression(o,t),l=i.value.getSubData(this.exec,n.postfix,t),u=new L(l.value,l.typeInfo);return l instanceof L&&a instanceof L&&(l.value=Math.max(l.value,a.value)),i.value instanceof Le&&i.value.setDataValue(this.exec,l,n.postfix,t),u}AtomicMin(e,t){let n=e.args[0];n instanceof Ue&&(n=n.right);const r=this.exec.getVariableName(n,t),i=t.getVariable(r);let o=e.args[1];const a=this.exec.evalExpression(o,t),l=i.value.getSubData(this.exec,n.postfix,t),u=new L(l.value,l.typeInfo);return l instanceof L&&a instanceof L&&(l.value=Math.min(l.value,a.value)),i.value instanceof Le&&i.value.setDataValue(this.exec,l,n.postfix,t),u}AtomicAnd(e,t){let n=e.args[0];n instanceof Ue&&(n=n.right);const r=this.exec.getVariableName(n,t),i=t.getVariable(r);let o=e.args[1];const a=this.exec.evalExpression(o,t),l=i.value.getSubData(this.exec,n.postfix,t),u=new L(l.value,l.typeInfo);return l instanceof L&&a instanceof L&&(l.value=l.value&a.value),i.value instanceof Le&&i.value.setDataValue(this.exec,l,n.postfix,t),u}AtomicOr(e,t){let n=e.args[0];n instanceof Ue&&(n=n.right);const r=this.exec.getVariableName(n,t),i=t.getVariable(r);let o=e.args[1];const a=this.exec.evalExpression(o,t),l=i.value.getSubData(this.exec,n.postfix,t),u=new L(l.value,l.typeInfo);return l instanceof L&&a instanceof L&&(l.value=l.value|a.value),i.value instanceof Le&&i.value.setDataValue(this.exec,l,n.postfix,t),u}AtomicXor(e,t){let n=e.args[0];n instanceof Ue&&(n=n.right);const r=this.exec.getVariableName(n,t),i=t.getVariable(r);let o=e.args[1];const a=this.exec.evalExpression(o,t),l=i.value.getSubData(this.exec,n.postfix,t),u=new L(l.value,l.typeInfo);return l instanceof L&&a instanceof L&&(l.value=l.value^a.value),i.value instanceof Le&&i.value.setDataValue(this.exec,l,n.postfix,t),u}AtomicExchange(e,t){let n=e.args[0];n instanceof Ue&&(n=n.right);const r=this.exec.getVariableName(n,t),i=t.getVariable(r);let o=e.args[1];const a=this.exec.evalExpression(o,t),l=i.value.getSubData(this.exec,n.postfix,t),u=new L(l.value,l.typeInfo);return l instanceof L&&a instanceof L&&(l.value=a.value),i.value instanceof Le&&i.value.setDataValue(this.exec,l,n.postfix,t),u}AtomicCompareExchangeWeak(e,t){return console.error("TODO: atomicCompareExchangeWeak"),null}Pack4x8snorm(e,t){return console.error("TODO: pack4x8snorm"),null}Pack4x8unorm(e,t){return console.error("TODO: pack4x8unorm"),null}Pack4xI8(e,t){return console.error("TODO: pack4xI8"),null}Pack4xU8(e,t){return console.error("TODO: pack4xU8"),null}Pack4x8Clamp(e,t){return console.error("TODO: pack4x8Clamp"),null}Pack4xU8Clamp(e,t){return console.error("TODO: pack4xU8Clamp"),null}Pack2x16snorm(e,t){return console.error("TODO: pack2x16snorm"),null}Pack2x16unorm(e,t){return console.error("TODO: pack2x16unorm"),null}Pack2x16float(e,t){return console.error("TODO: pack2x16float"),null}Unpack4x8snorm(e,t){return console.error("TODO: unpack4x8snorm"),null}Unpack4x8unorm(e,t){return console.error("TODO: unpack4x8unorm"),null}Unpack4xI8(e,t){return console.error("TODO: unpack4xI8"),null}Unpack4xU8(e,t){return console.error("TODO: unpack4xU8"),null}Unpack2x16snorm(e,t){return console.error("TODO: unpack2x16snorm"),null}Unpack2x16unorm(e,t){return console.error("TODO: unpack2x16unorm"),null}Unpack2x16float(e,t){return console.error("TODO: unpack2x16float"),null}StorageBarrier(e,t){return null}TextureBarrier(e,t){return null}WorkgroupBarrier(e,t){return null}WorkgroupUniformLoad(e,t){return null}SubgroupAdd(e,t){return console.error("TODO: subgroupAdd"),null}SubgroupExclusiveAdd(e,t){return console.error("TODO: subgroupExclusiveAdd"),null}SubgroupInclusiveAdd(e,t){return console.error("TODO: subgroupInclusiveAdd"),null}SubgroupAll(e,t){return console.error("TODO: subgroupAll"),null}SubgroupAnd(e,t){return console.error("TODO: subgroupAnd"),null}SubgroupAny(e,t){return console.error("TODO: subgroupAny"),null}SubgroupBallot(e,t){return console.error("TODO: subgroupBallot"),null}SubgroupBroadcast(e,t){return console.error("TODO: subgroupBroadcast"),null}SubgroupBroadcastFirst(e,t){return console.error("TODO: subgroupBroadcastFirst"),null}SubgroupElect(e,t){return console.error("TODO: subgroupElect"),null}SubgroupMax(e,t){return console.error("TODO: subgroupMax"),null}SubgroupMin(e,t){return console.error("TODO: subgroupMin"),null}SubgroupMul(e,t){return console.error("TODO: subgroupMul"),null}SubgroupExclusiveMul(e,t){return console.error("TODO: subgroupExclusiveMul"),null}SubgroupInclusiveMul(e,t){return console.error("TODO: subgroupInclusiveMul"),null}SubgroupOr(e,t){return console.error("TODO: subgroupOr"),null}SubgroupShuffle(e,t){return console.error("TODO: subgroupShuffle"),null}SubgroupShuffleDown(e,t){return console.error("TODO: subgroupShuffleDown"),null}SubgroupShuffleUp(e,t){return console.error("TODO: subgroupShuffleUp"),null}SubgroupShuffleXor(e,t){return console.error("TODO: subgroupShuffleXor"),null}SubgroupXor(e,t){return console.error("TODO: subgroupXor"),null}QuadBroadcast(e,t){return console.error("TODO: quadBroadcast"),null}QuadSwapDiagonal(e,t){return console.error("TODO: quadSwapDiagonal"),null}QuadSwapX(e,t){return console.error("TODO: quadSwapX"),null}QuadSwapY(e,t){return console.error("TODO: quadSwapY"),null}}const na={vec2:2,vec2f:2,vec2i:2,vec2u:2,vec2b:2,vec2h:2,vec3:3,vec3f:3,vec3i:3,vec3u:3,vec3b:3,vec3h:3,vec4:4,vec4f:4,vec4i:4,vec4u:4,vec4b:4,vec4h:4},ft={mat2x2:[2,2,4],mat2x2f:[2,2,4],mat2x2h:[2,2,4],mat2x3:[2,3,6],mat2x3f:[2,3,6],mat2x3h:[2,3,6],mat2x4:[2,4,8],mat2x4f:[2,4,8],mat2x4h:[2,4,8],mat3x2:[3,2,6],mat3x2f:[3,2,6],mat3x2h:[3,2,6],mat3x3:[3,3,9],mat3x3f:[3,3,9],mat3x3h:[3,3,9],mat3x4:[3,4,12],mat3x4f:[3,4,12],mat3x4h:[3,4,12],mat4x2:[4,2,8],mat4x2f:[4,2,8],mat4x2h:[4,2,8],mat4x3:[4,3,12],mat4x3f:[4,3,12],mat4x3h:[4,3,12],mat4x4:[4,4,16],mat4x4f:[4,4,16],mat4x4h:[4,4,16]};class at extends Qk{constructor(e,t){var n;super(),this.ast=e??[],this.reflection=new yn,this.reflection.updateAST(this.ast),this.context=(n=t?.clone())!==null&&n!==void 0?n:new Xl,this.builtins=new Zk(this),this.typeInfo={bool:this.getTypeInfo(K.bool),i32:this.getTypeInfo(K.i32),u32:this.getTypeInfo(K.u32),f32:this.getTypeInfo(K.f32),f16:this.getTypeInfo(K.f16),vec2f:this.getTypeInfo(U.vec2f),vec2u:this.getTypeInfo(U.vec2u),vec2i:this.getTypeInfo(U.vec2i),vec2h:this.getTypeInfo(U.vec2h),vec3f:this.getTypeInfo(U.vec3f),vec3u:this.getTypeInfo(U.vec3u),vec3i:this.getTypeInfo(U.vec3i),vec3h:this.getTypeInfo(U.vec3h),vec4f:this.getTypeInfo(U.vec4f),vec4u:this.getTypeInfo(U.vec4u),vec4i:this.getTypeInfo(U.vec4i),vec4h:this.getTypeInfo(U.vec4h),mat2x2f:this.getTypeInfo(U.mat2x2f),mat2x3f:this.getTypeInfo(U.mat2x3f),mat2x4f:this.getTypeInfo(U.mat2x4f),mat3x2f:this.getTypeInfo(U.mat3x2f),mat3x3f:this.getTypeInfo(U.mat3x3f),mat3x4f:this.getTypeInfo(U.mat3x4f),mat4x2f:this.getTypeInfo(U.mat4x2f),mat4x3f:this.getTypeInfo(U.mat4x3f),mat4x4f:this.getTypeInfo(U.mat4x4f)}}getVariableValue(e){var t,n;const r=(n=(t=this.context.getVariable(e))===null||t===void 0?void 0:t.value)!==null&&n!==void 0?n:null;if(r===null)return null;if(r instanceof L)return r.value;if(r instanceof M||r instanceof ue)return Array.from(r.data);if(r instanceof Le&&r.typeInfo instanceof Vn){if(r.typeInfo.format.name==="u32")return Array.from(new Uint32Array(r.buffer,r.offset,r.typeInfo.count));if(r.typeInfo.format.name==="i32")return Array.from(new Int32Array(r.buffer,r.offset,r.typeInfo.count));if(r.typeInfo.format.name==="f32")return Array.from(new Float32Array(r.buffer,r.offset,r.typeInfo.count))}return console.error(`Unsupported return variable type ${r.typeInfo.name}`),null}execute(e){(e=e??{}).constants&&this._setOverrides(e.constants,this.context),this._execStatements(this.ast,this.context)}dispatchWorkgroups(e,t,n,r){const i=this.context.clone();(r=r??{}).constants&&this._setOverrides(r.constants,i),this._execStatements(this.ast,i);const o=i.getFunction(e);if(!o)return void console.error(`Function ${e} not found`);if(typeof t=="number")t=[t,1,1];else{if(t.length===0)return void console.error("Invalid dispatch count");t.length===1?t=[t[0],1,1]:t.length===2?t=[t[0],t[1],1]:t.length>3&&(t=[t[0],t[1],t[2]])}const a=t[0],l=t[1],u=t[2],c=this.getTypeInfo("vec3u");i.setVariable("@num_workgroups",new M(t,c));const h=this.reflection.getFunctionInfo(e);h===null&&console.error(`Function ${e} not found in reflection data`);for(const d in n)for(const w in n[d]){const I=n[d][w];i.variables.forEach(E=>{var m;const S=E.node;if(S?.attributes){let b=null,f=null;for(const _ of S.attributes)_.name==="binding"?b=_.value:_.name==="group"&&(f=_.value);if(w==b&&d==f){let _=!1;for(const v of h.resources)if(v.name===E.name&&v.group===parseInt(d)&&v.binding===parseInt(w)){_=!0;break}if(_)if(I.texture!==void 0&&I.descriptor!==void 0){const v=new Tn(I.texture,this.getTypeInfo(S.type),I.descriptor,(m=I.texture.view)!==null&&m!==void 0?m:null);E.value=v}else I.uniform!==void 0?E.value=new Le(I.uniform,this.getTypeInfo(S.type)):E.value=new Le(I,this.getTypeInfo(S.type))}}})}for(let d=0;d<u;++d)for(let w=0;w<l;++w)for(let I=0;I<a;++I)i.setVariable("@workgroup_id",new M([I,w,d],this.getTypeInfo("vec3u"))),this._dispatchWorkgroup(o,[I,w,d],i)}execStatement(e,t){if(e instanceof zp)return this.evalExpression(e.value,t);if(e instanceof Gp){if(e.condition){const n=this.evalExpression(e.condition,t);if(!(n instanceof L))throw new Error("Invalid break-if condition");if(!n.value)return null}return at._breakObj}if(e instanceof Wp)return at._continueObj;if(e instanceof mr)this._let(e,t);else if(e instanceof Cn)this._var(e,t);else if(e instanceof Si)this._const(e,t);else if(e instanceof kr)this._function(e,t);else{if(e instanceof Up)return this._if(e,t);if(e instanceof Fp)return this._switch(e,t);if(e instanceof Pp)return this._for(e,t);if(e instanceof Mp)return this._while(e,t);if(e instanceof Bp)return this._loop(e,t);if(e instanceof La){const n=t.clone();return n.currentFunctionName=t.currentFunctionName,this._execStatements(e.body,n)}if(e instanceof Lp)this._assign(e,t);else if(e instanceof Rp)this._increment(e,t);else{if(e instanceof kn)return null;if(e instanceof Vl){const n=e.name;t.getVariable(n)===null&&t.setVariable(n,new L(0,this.getTypeInfo("u32")))}else if(e instanceof Gl)this._call(e,t);else{if(e instanceof Vp||e instanceof Wl)return null;console.error("Invalid statement type.",e,`Line ${e.line}`)}}}return null}evalExpression(e,t){return e instanceof Jt?this._evalBinaryOp(e,t):e instanceof We?this._evalLiteral(e,t):e instanceof Mt?this._evalVariable(e,t):e instanceof ql?this._evalCall(e,t):e instanceof fn?this._evalCreate(e,t):e instanceof qp?this._evalConst(e,t):e instanceof Hp?this._evalBitcast(e,t):e instanceof Ue?this._evalUnaryOp(e,t):(console.error("Invalid expression type",e,`Line ${e.line}`),null)}getTypeInfo(e){var t;if(e instanceof K){const r=this.reflection.getTypeInfo(e);if(r!==null)return r}let n=(t=this.typeInfo[e])!==null&&t!==void 0?t:null;return n!==null||(n=this.reflection.getTypeInfoByName(e)),n}_setOverrides(e,t){for(const n in e){const r=e[n],i=this.reflection.getOverrideInfo(n);i!==null?(i.type===null&&(i.type=this.getTypeInfo("u32")),i.type.name==="u32"||i.type.name==="i32"||i.type.name==="f32"||i.type.name==="f16"?t.setVariable(n,new L(r,i.type)):i.type.name==="bool"?t.setVariable(n,new L(r?1:0,i.type)):i.type.name==="vec2"||i.type.name==="vec3"||i.type.name==="vec4"||i.type.name==="vec2f"||i.type.name==="vec3f"||i.type.name==="vec4f"||i.type.name==="vec2i"||i.type.name==="vec3i"||i.type.name==="vec4i"||i.type.name==="vec2u"||i.type.name==="vec3u"||i.type.name==="vec4u"||i.type.name==="vec2h"||i.type.name==="vec3h"||i.type.name==="vec4h"?t.setVariable(n,new M(r,i.type)):console.error(`Invalid constant type for ${n}`)):console.error(`Override ${n} does not exist in the shader.`)}}_dispatchWorkgroup(e,t,n){const r=[1,1,1];for(const c of e.node.attributes)if(c.name==="workgroup_size"){if(c.value.length>0){const h=n.getVariableValue(c.value[0]);r[0]=h instanceof L?h.value:parseInt(c.value[0])}if(c.value.length>1){const h=n.getVariableValue(c.value[1]);r[1]=h instanceof L?h.value:parseInt(c.value[1])}if(c.value.length>2){const h=n.getVariableValue(c.value[2]);r[2]=h instanceof L?h.value:parseInt(c.value[2])}}const i=this.getTypeInfo("vec3u"),o=this.getTypeInfo("u32");n.setVariable("@workgroup_size",new M(r,i));const a=r[0],l=r[1],u=r[2];for(let c=0,h=0;c<u;++c)for(let d=0;d<l;++d)for(let w=0;w<a;++w,++h){const I=[w,d,c],E=[w+t[0]*r[0],d+t[1]*r[1],c+t[2]*r[2]];n.setVariable("@local_invocation_id",new M(I,i)),n.setVariable("@global_invocation_id",new M(E,i)),n.setVariable("@local_invocation_index",new L(h,o)),this._dispatchExec(e,n)}}_dispatchExec(e,t){for(const n of e.node.args)for(const r of n.attributes)if(r.name==="builtin"){const i=`@${r.value}`,o=t.getVariable(i);o!==void 0&&t.variables.set(n.name,o)}this._execStatements(e.node.body,t)}getVariableName(e,t){for(;e instanceof Ue;)e=e.right;return e instanceof Mt?e.name:(console.error("Unknown variable type",e,"Line",e.line),null)}_execStatements(e,t){for(const n of e){if(n instanceof Array){const i=t.clone(),o=this._execStatements(n,i);if(o)return o;continue}const r=this.execStatement(n,t);if(r)return r}return null}_call(e,t){const n=t.clone();n.currentFunctionName=e.name;const r=t.getFunction(e.name);if(r){for(let i=0;i<r.node.args.length;++i){const o=r.node.args[i],a=this.evalExpression(e.args[i],n);n.setVariable(o.name,a,o)}this._execStatements(r.node.body,n)}else e.isBuiltin?this._callBuiltinFunction(e,n):this.getTypeInfo(e.name)&&this._evalCreate(e,t)}_increment(e,t){const n=this.getVariableName(e.variable,t),r=t.getVariable(n);r?e.operator==="++"?r.value instanceof L?r.value.value++:console.error(`Variable ${n} is not a scalar. Line ${e.line}`):e.operator==="--"?r.value instanceof L?r.value.value--:console.error(`Variable ${n} is not a scalar. Line ${e.line}`):console.error(`Unknown increment operator ${e.operator}. Line ${e.line}`):console.error(`Variable ${n} not found. Line ${e.line}`)}_getVariableData(e,t){if(e instanceof Mt){const n=this.getVariableName(e,t),r=t.getVariable(n);return r===null?(console.error(`Variable ${n} not found. Line ${e.line}`),null):r.value.getSubData(this,e.postfix,t)}if(e instanceof Ue){if(e.operator==="*"){const n=this._getVariableData(e.right,t);return n instanceof xs?n.reference.getSubData(this,e.postfix,t):(console.error(`Variable ${e.right} is not a pointer. Line ${e.line}`),null)}if(e.operator==="&"){const n=this._getVariableData(e.right,t);return new xs(n)}}return null}_assign(e,t){let n=null,r="<var>",i=null;if(e.variable instanceof Ue){const l=this._getVariableData(e.variable,t),u=this.evalExpression(e.value,t),c=e.operator;if(c==="="){if(l instanceof L||l instanceof M||l instanceof ue){if(u instanceof L||u instanceof M||u instanceof ue&&l.data.length===u.data.length)return void l.data.set(u.data);console.error(`Invalid assignment. Line ${e.line}`)}else if(l instanceof Le&&u instanceof Le&&l.buffer.byteLength-l.offset>=u.buffer.byteLength-u.offset)return void(l.buffer.byteLength%4==0?new Uint32Array(l.buffer,l.offset,l.typeInfo.size/4).set(new Uint32Array(u.buffer,u.offset,u.typeInfo.size/4)):new Uint8Array(l.buffer,l.offset,l.typeInfo.size).set(new Uint8Array(u.buffer,u.offset,u.typeInfo.size)));return console.error(`Invalid assignment. Line ${e.line}`),null}if(c==="+=")return l instanceof L||l instanceof M||l instanceof ue?u instanceof L||u instanceof M||u instanceof ue?void l.data.set(u.data.map((h,d)=>l.data[d]+h)):void console.error(`Invalid assignment . Line ${e.line}`):void console.error(`Invalid assignment. Line ${e.line}`);if(c==="-=")return(l instanceof L||l instanceof M||l instanceof ue)&&(u instanceof L||u instanceof M||u instanceof ue)?void l.data.set(u.data.map((h,d)=>l.data[d]-h)):void console.error(`Invalid assignment. Line ${e.line}`)}if(e.variable instanceof Ue){if(e.variable.operator==="*"){r=this.getVariableName(e.variable.right,t);const l=t.getVariable(r);if(!(l&&l.value instanceof xs))return void console.error(`Variable ${r} is not a pointer. Line ${e.line}`);n=l.value.reference;let u=e.variable.postfix;if(!u){let c=e.variable.right;for(;c instanceof Ue;){if(c.postfix){u=c.postfix;break}c=c.right}}u&&(n=n.getSubData(this,u,t))}}else{i=e.variable.postfix,r=this.getVariableName(e.variable,t);const l=t.getVariable(r);if(l===null)return void console.error(`Variable ${r} not found. Line ${e.line}`);n=l.value}if(n instanceof xs&&(n=n.reference),n===null)return void console.error(`Variable ${r} not found. Line ${e.line}`);const o=this.evalExpression(e.value,t),a=e.operator;if(a!=="="){const l=n.getSubData(this,i,t);if(l instanceof M&&o instanceof L){const u=l.data,c=o.value;if(a==="+=")for(let h=0;h<u.length;++h)u[h]+=c;else if(a==="-=")for(let h=0;h<u.length;++h)u[h]-=c;else if(a==="*=")for(let h=0;h<u.length;++h)u[h]*=c;else if(a==="/=")for(let h=0;h<u.length;++h)u[h]/=c;else if(a==="%=")for(let h=0;h<u.length;++h)u[h]%=c;else if(a==="&=")for(let h=0;h<u.length;++h)u[h]&=c;else if(a==="|=")for(let h=0;h<u.length;++h)u[h]|=c;else if(a==="^=")for(let h=0;h<u.length;++h)u[h]^=c;else if(a==="<<=")for(let h=0;h<u.length;++h)u[h]<<=c;else if(a===">>=")for(let h=0;h<u.length;++h)u[h]>>=c;else console.error(`Invalid operator ${a}. Line ${e.line}`)}else if(l instanceof M&&o instanceof M){const u=l.data,c=o.data;if(u.length!==c.length)return void console.error(`Vector length mismatch. Line ${e.line}`);if(a==="+=")for(let h=0;h<u.length;++h)u[h]+=c[h];else if(a==="-=")for(let h=0;h<u.length;++h)u[h]-=c[h];else if(a==="*=")for(let h=0;h<u.length;++h)u[h]*=c[h];else if(a==="/=")for(let h=0;h<u.length;++h)u[h]/=c[h];else if(a==="%=")for(let h=0;h<u.length;++h)u[h]%=c[h];else if(a==="&=")for(let h=0;h<u.length;++h)u[h]&=c[h];else if(a==="|=")for(let h=0;h<u.length;++h)u[h]|=c[h];else if(a==="^=")for(let h=0;h<u.length;++h)u[h]^=c[h];else if(a==="<<=")for(let h=0;h<u.length;++h)u[h]<<=c[h];else if(a===">>=")for(let h=0;h<u.length;++h)u[h]>>=c[h];else console.error(`Invalid operator ${a}. Line ${e.line}`)}else{if(!(l instanceof L&&o instanceof L))return void console.error(`Invalid type for ${e.operator} operator. Line ${e.line}`);a==="+="?l.value+=o.value:a==="-="?l.value-=o.value:a==="*="?l.value*=o.value:a==="/="?l.value/=o.value:a==="%="?l.value%=o.value:a==="&="?l.value&=o.value:a==="|="?l.value|=o.value:a==="^="?l.value^=o.value:a==="<<="?l.value<<=o.value:a===">>="?l.value>>=o.value:console.error(`Invalid operator ${a}. Line ${e.line}`)}return void(n instanceof Le&&n.setDataValue(this,l,i,t))}if(n instanceof Le)n.setDataValue(this,o,i,t);else if(i){if(!(n instanceof M||n instanceof ue))return void console.error(`Variable ${r} is not a vector or matrix. Line ${e.line}`);if(i instanceof Ls){const l=this.evalExpression(i.index,t).value;if(n instanceof M){if(!(o instanceof L))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[l]=o.value}else{if(!(n instanceof ue))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);{const u=this.evalExpression(i.index,t).value;if(u<0)return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);if(!(o instanceof M))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);{const c=n.typeInfo.getTypeName();if(c==="mat2x2"||c==="mat2x2f"||c==="mat2x2h"){if(!(u<2&&o.data.length===2))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[2*u]=o.data[0],n.data[2*u+1]=o.data[1]}else if(c==="mat2x3"||c==="mat2x3f"||c==="mat2x3h"){if(!(u<2&&o.data.length===3))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[3*u]=o.data[0],n.data[3*u+1]=o.data[1],n.data[3*u+2]=o.data[2]}else if(c==="mat2x4"||c==="mat2x4f"||c==="mat2x4h"){if(!(u<2&&o.data.length===4))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[4*u]=o.data[0],n.data[4*u+1]=o.data[1],n.data[4*u+2]=o.data[2],n.data[4*u+3]=o.data[3]}else if(c==="mat3x2"||c==="mat3x2f"||c==="mat3x2h"){if(!(u<3&&o.data.length===2))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[2*u]=o.data[0],n.data[2*u+1]=o.data[1]}else if(c==="mat3x3"||c==="mat3x3f"||c==="mat3x3h"){if(!(u<3&&o.data.length===3))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[3*u]=o.data[0],n.data[3*u+1]=o.data[1],n.data[3*u+2]=o.data[2]}else if(c==="mat3x4"||c==="mat3x4f"||c==="mat3x4h"){if(!(u<3&&o.data.length===4))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[4*u]=o.data[0],n.data[4*u+1]=o.data[1],n.data[4*u+2]=o.data[2],n.data[4*u+3]=o.data[3]}else if(c==="mat4x2"||c==="mat4x2f"||c==="mat4x2h"){if(!(u<4&&o.data.length===2))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[2*u]=o.data[0],n.data[2*u+1]=o.data[1]}else if(c==="mat4x3"||c==="mat4x3f"||c==="mat4x3h"){if(!(u<4&&o.data.length===3))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[3*u]=o.data[0],n.data[3*u+1]=o.data[1],n.data[3*u+2]=o.data[2]}else{if(c!=="mat4x4"&&c!=="mat4x4f"&&c!=="mat4x4h")return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);if(!(u<4&&o.data.length===4))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);n.data[4*u]=o.data[0],n.data[4*u+1]=o.data[1],n.data[4*u+2]=o.data[2],n.data[4*u+3]=o.data[3]}}}}}else if(i instanceof ms){const l=i.value;if(!(n instanceof M))return void console.error(`Invalid assignment to ${l}. Variable ${r} is not a vector. Line ${e.line}`);if(o instanceof L){if(l.length>1)return void console.error(`Invalid assignment to ${l} for variable ${r}. Line ${e.line}`);if(l==="x")n.data[0]=o.value;else if(l==="y"){if(n.data.length<2)return void console.error(`Invalid assignment to ${l} for variable ${r}. Line ${e.line}`);n.data[1]=o.value}else if(l==="z"){if(n.data.length<3)return void console.error(`Invalid assignment to ${l} for variable ${r}. Line ${e.line}`);n.data[2]=o.value}else if(l==="w"){if(n.data.length<4)return void console.error(`Invalid assignment to ${l} for variable ${r}. Line ${e.line}`);n.data[3]=o.value}}else{if(!(o instanceof M))return void console.error(`Invalid assignment to ${r}. Line ${e.line}`);if(l.length!==o.data.length)return void console.error(`Invalid assignment to ${l} for variable ${r}. Line ${e.line}`);for(let u=0;u<l.length;++u){const c=l[u];if(c==="x"||c==="r")n.data[0]=o.data[u];else if(c==="y"||c==="g"){if(o.data.length<2)return void console.error(`Invalid assignment to ${c} for variable ${r}. Line ${e.line}`);n.data[1]=o.data[u]}else if(c==="z"||c==="b"){if(o.data.length<3)return void console.error(`Invalid assignment to ${c} for variable ${r}. Line ${e.line}`);n.data[2]=o.data[u]}else{if(c!=="w"&&c!=="a")return void console.error(`Invalid assignment to ${c} for variable ${r}. Line ${e.line}`);if(o.data.length<4)return void console.error(`Invalid assignment to ${c} for variable ${r}. Line ${e.line}`);n.data[3]=o.data[u]}}}}}else n instanceof L&&o instanceof L?n.value=o.value:n instanceof M&&o instanceof M||n instanceof ue&&o instanceof ue?n.data.set(o.data):console.error(`Invalid assignment to ${r}. Line ${e.line}`)}_function(e,t){const n=new Kl(e);t.functions.set(e.name,n)}_const(e,t){let n=null;e.value!==null&&(n=this.evalExpression(e.value,t)),t.createVariable(e.name,n,e)}_let(e,t){let n=null;if(e.value!==null){if(n=this.evalExpression(e.value,t),n===null)return void console.error(`Invalid value for variable ${e.name}. Line ${e.line}`);e.value instanceof Ue||(n=n.clone())}else{const r=e.type.name;if(r==="f32"||r==="i32"||r==="u32"||r==="bool"||r==="f16"||r==="vec2"||r==="vec3"||r==="vec4"||r==="vec2f"||r==="vec3f"||r==="vec4f"||r==="vec2i"||r==="vec3i"||r==="vec4i"||r==="vec2u"||r==="vec3u"||r==="vec4u"||r==="vec2h"||r==="vec3h"||r==="vec4h"||r==="vec2b"||r==="vec3b"||r==="vec4b"||r==="mat2x2"||r==="mat2x3"||r==="mat2x4"||r==="mat3x2"||r==="mat3x3"||r==="mat3x4"||r==="mat4x2"||r==="mat4x3"||r==="mat4x4"||r==="mat2x2f"||r==="mat2x3f"||r==="mat2x4f"||r==="mat3x2f"||r==="mat3x3f"||r==="mat3x4f"||r==="mat4x2f"||r==="mat4x3f"||r==="mat4x4f"||r==="mat2x2h"||r==="mat2x3h"||r==="mat2x4h"||r==="mat3x2h"||r==="mat3x3h"||r==="mat3x4h"||r==="mat4x2h"||r==="mat4x3h"||r==="mat4x4h"||r==="array"){const i=new fn(e.type,[]);n=this._evalCreate(i,t)}}t.createVariable(e.name,n,e)}_var(e,t){let n=null;if(e.value!==null){if(n=this.evalExpression(e.value,t),n===null)return void console.error(`Invalid value for variable ${e.name}. Line ${e.line}`);e.value instanceof Ue||(n=n.clone())}else{if(e.type===null)return void console.error(`Variable ${e.name} has no type. Line ${e.line}`);const r=e.type.name;if(r==="f32"||r==="i32"||r==="u32"||r==="bool"||r==="f16"||r==="vec2"||r==="vec3"||r==="vec4"||r==="vec2f"||r==="vec3f"||r==="vec4f"||r==="vec2i"||r==="vec3i"||r==="vec4i"||r==="vec2u"||r==="vec3u"||r==="vec4u"||r==="vec2h"||r==="vec3h"||r==="vec4h"||r==="vec2b"||r==="vec3b"||r==="vec4b"||r==="mat2x2"||r==="mat2x3"||r==="mat2x4"||r==="mat3x2"||r==="mat3x3"||r==="mat3x4"||r==="mat4x2"||r==="mat4x3"||r==="mat4x4"||r==="mat2x2f"||r==="mat2x3f"||r==="mat2x4f"||r==="mat3x2f"||r==="mat3x3f"||r==="mat3x4f"||r==="mat4x2f"||r==="mat4x3f"||r==="mat4x4f"||r==="mat2x2h"||r==="mat2x3h"||r==="mat2x4h"||r==="mat3x2h"||r==="mat3x3h"||r==="mat3x4h"||r==="mat4x2h"||r==="mat4x3h"||r==="mat4x4h"||e.type instanceof gr||e.type instanceof kn||e.type instanceof U){const i=new fn(e.type,[]);n=this._evalCreate(i,t)}}t.createVariable(e.name,n,e)}_switch(e,t){t=t.clone();const n=this.evalExpression(e.condition,t);if(!(n instanceof L))return console.error(`Invalid if condition. Line ${e.line}`),null;let r=null;for(const i of e.cases)if(i instanceof Xp)for(const o of i.selectors){if(o instanceof ki){r=i;continue}const a=this.evalExpression(o,t);if(!(a instanceof L))return console.error(`Invalid case selector. Line ${e.line}`),null;if(a.value===n.value)return this._execStatements(i.body,t)}else i instanceof Yp&&(r=i);return r?this._execStatements(r.body,t):null}_if(e,t){t=t.clone();const n=this.evalExpression(e.condition,t);if(!(n instanceof L))return console.error(`Invalid if condition. Line ${e.line}`),null;if(n.value)return this._execStatements(e.body,t);for(const r of e.elseif){const i=this.evalExpression(r.condition,t);if(!(i instanceof L))return console.error(`Invalid if condition. Line ${e.line}`),null;if(i.value)return this._execStatements(r.body,t)}return e.else?this._execStatements(e.else,t):null}_getScalarValue(e){return e instanceof L?e.value:(console.error("Expected scalar value.",e),0)}_for(e,t){for(t=t.clone(),this.execStatement(e.init,t);this._getScalarValue(this.evalExpression(e.condition,t));){const n=this._execStatements(e.body,t);if(n===at._breakObj)break;if(n!==null&&n!==at._continueObj)return n;this.execStatement(e.increment,t)}return null}_loop(e,t){for(t=t.clone();;){const n=this._execStatements(e.body,t);if(n===at._breakObj)break;if(n===at._continueObj){if(e.continuing&&this._execStatements(e.continuing.body,t)===at._breakObj)break}else if(n!==null)return n}return null}_while(e,t){for(t=t.clone();this._getScalarValue(this.evalExpression(e.condition,t));){const n=this._execStatements(e.body,t);if(n===at._breakObj)break;if(n!==at._continueObj&&n!==null)return n}return null}_evalBitcast(e,t){const n=this.evalExpression(e.value,t),r=e.type;if(n instanceof L){const i=Fc(n.value,n.typeInfo.name,r.name);return new L(i,this.getTypeInfo(r))}if(n instanceof M){const i=n.typeInfo.getTypeName();let o="";if(i.endsWith("f"))o="f32";else if(i.endsWith("i"))o="i32";else if(i.endsWith("u"))o="u32";else if(i.endsWith("b"))o="bool";else{if(!i.endsWith("h"))return console.error(`Unknown vector type ${i}. Line ${e.line}`),null;o="f16"}const a=r.getTypeName();let l="";if(a.endsWith("f"))l="f32";else if(a.endsWith("i"))l="i32";else if(a.endsWith("u"))l="u32";else if(a.endsWith("b"))l="bool";else{if(!a.endsWith("h"))return console.error(`Unknown vector type ${l}. Line ${e.line}`),null;l="f16"}const u=((c,h,d)=>{if(h===d)return c;const w=new Array(c.length);for(let I=0;I<c.length;I++)w[I]=Fc(c[I],h,d);return w})(Array.from(n.data),o,l);return new M(u,this.getTypeInfo(r))}return console.error(`TODO: bitcast for ${n.typeInfo.name}. Line ${e.line}`),null}_evalConst(e,t){return t.getVariableValue(e.name).clone().getSubData(this,e.postfix,t)}_evalCreate(e,t){var n;if(e instanceof fn){if(e.type===null)return Ba.void;switch(e.type.getTypeName()){case"bool":case"i32":case"u32":case"f32":case"f16":return this._callConstructorValue(e,t);case"vec2":case"vec3":case"vec4":case"vec2f":case"vec3f":case"vec4f":case"vec2h":case"vec3h":case"vec4h":case"vec2i":case"vec3i":case"vec4i":case"vec2u":case"vec3u":case"vec4u":case"vec2b":case"vec3b":case"vec4b":return this._callConstructorVec(e,t);case"mat2x2":case"mat2x2f":case"mat2x2h":case"mat2x3":case"mat2x3f":case"mat2x3h":case"mat2x4":case"mat2x4f":case"mat2x4h":case"mat3x2":case"mat3x2f":case"mat3x2h":case"mat3x3":case"mat3x3f":case"mat3x3h":case"mat3x4":case"mat3x4f":case"mat3x4h":case"mat4x2":case"mat4x2f":case"mat4x2h":case"mat4x3":case"mat4x3f":case"mat4x3h":case"mat4x4":case"mat4x4f":case"mat4x4h":return this._callConstructorMatrix(e,t)}}const r=e instanceof fn?e.type.name:e.name,i=e instanceof fn?this.getTypeInfo(e.type):this.getTypeInfo(e.name);if(i===null)return console.error(`Unknown type ${r}. Line ${e.line}`),null;if(i.size===0)return null;const o=new Le(new ArrayBuffer(i.size),i,0);if(i instanceof Ln){if(e.args)for(let a=0;a<e.args.length;++a){const l=i.members[a],u=e.args[a],c=this.evalExpression(u,t);o.setData(this,c,l.type,l.offset,t)}}else if(i instanceof Vn){let a=0;if(e.args)for(let l=0;l<e.args.length;++l){const u=e.args[l],c=this.evalExpression(u,t);i.format===null&&(((n=c.typeInfo)===null||n===void 0?void 0:n.name)==="x32"?i.format=this.getTypeInfo("i32"):i.format=c.typeInfo),o.setData(this,c,i.format,a,t),a+=i.stride}}else console.error(`Unknown type "${r}". Line ${e.line}`);return e instanceof fn?o.getSubData(this,e.postfix,t):o}_evalLiteral(e,t){const n=this.getTypeInfo(e.type),r=n.name;return r==="x32"||r==="u32"||r==="f32"||r==="f16"||r==="i32"||r==="bool"?new L(e.scalarValue,n):r==="vec2"||r==="vec3"||r==="vec4"||r==="vec2f"||r==="vec3f"||r==="vec4f"||r==="vec2h"||r==="vec3h"||r==="vec4h"||r==="vec2i"||r==="vec3i"||r==="vec4i"||r==="vec2u"||r==="vec3u"||r==="vec4u"?this._callConstructorVec(e,t):r==="mat2x2"||r==="mat2x3"||r==="mat2x4"||r==="mat3x2"||r==="mat3x3"||r==="mat3x4"||r==="mat4x2"||r==="mat4x3"||r==="mat4x4"||r==="mat2x2f"||r==="mat2x3f"||r==="mat2x4f"||r==="mat3x2f"||r==="mat3x3f"||r==="mat3x4f"||r==="mat4x2f"||r==="mat4x3f"||r==="mat4x4f"||r==="mat2x2h"||r==="mat2x3h"||r==="mat2x4h"||r==="mat3x2h"||r==="mat3x3h"||r==="mat3x4h"||r==="mat4x2h"||r==="mat4x3h"||r==="mat4x4h"?this._callConstructorMatrix(e,t):e.value}_evalVariable(e,t){const n=t.getVariableValue(e.name);return n===null?n:n.getSubData(this,e.postfix,t)}_maxFormatTypeInfo(e){let t=e[0];if(t.name==="f32")return t;for(let n=1;n<e.length;++n){const r=at._priority.get(t.name);at._priority.get(e[n].name)<r&&(t=e[n])}return t.name==="x32"?this.getTypeInfo("i32"):t}_evalUnaryOp(e,t){const n=this.evalExpression(e.right,t);if(e.operator==="&")return new xs(n);if(e.operator==="*")return n instanceof xs?n.reference.getSubData(this,e.postfix,t):(console.error(`Invalid dereference. Line ${e.line}`),null);const r=n instanceof L?n.value:n instanceof M?Array.from(n.data):null;switch(e.operator){case"+":{if(ee(r)){const a=r.map((l,u)=>+l);return new M(a,n.typeInfo)}const i=r,o=this._maxFormatTypeInfo([n.typeInfo,n.typeInfo]);return new L(+i,o)}case"-":{if(ee(r)){const a=r.map((l,u)=>-l);return new M(a,n.typeInfo)}const i=r,o=this._maxFormatTypeInfo([n.typeInfo,n.typeInfo]);return new L(-i,o)}case"!":{if(ee(r)){const a=r.map((l,u)=>l?0:1);return new M(a,n.typeInfo)}const i=r,o=this._maxFormatTypeInfo([n.typeInfo,n.typeInfo]);return new L(i?0:1,o)}case"~":{if(ee(r)){const a=r.map((l,u)=>~l);return new M(a,n.typeInfo)}const i=r,o=this._maxFormatTypeInfo([n.typeInfo,n.typeInfo]);return new L(~i,o)}}return console.error(`Invalid unary operator ${e.operator}. Line ${e.line}`),null}_evalBinaryOp(e,t){const n=this.evalExpression(e.left,t),r=this.evalExpression(e.right,t),i=n instanceof L?n.value:n instanceof M||n instanceof ue?Array.from(n.data):null,o=r instanceof L?r.value:r instanceof M||r instanceof ue?Array.from(r.data):null;switch(e.operator){case"+":{if(ee(i)&&ee(o)){const c=i,h=o;if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,I)=>w+h[I]);return new M(d,n.typeInfo)}if(ee(i)){const c=o,h=i.map((d,w)=>d+c);return new M(h,n.typeInfo)}if(ee(o)){const c=i,h=o.map((d,w)=>c+d);return new M(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new L(a+l,u)}case"-":{if(ee(i)&&ee(o)){const c=i,h=o;if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,I)=>w-h[I]);return new M(d,n.typeInfo)}if(ee(i)){const c=o,h=i.map((d,w)=>d-c);return new M(h,n.typeInfo)}if(ee(o)){const c=i,h=o.map((d,w)=>c-d);return new M(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new L(a-l,u)}case"*":{if(ee(i)&&ee(o)){const c=i,h=o;if(n instanceof ue&&r instanceof ue){const d=((m,S,b,f)=>{if(ft[S.name]===void 0||ft[f.name]===void 0)return null;const _=ft[S.name][0],v=ft[S.name][1],T=ft[f.name][0];if(_!==ft[f.name][1])return null;const N=new Array(T*v);for(let O=0;O<v;O++)for(let $=0;$<T;$++){let A=0;for(let g=0;g<_;g++)A+=m[g*v+O]*b[$*_+g];N[O*T+$]=A}return N})(c,n.typeInfo,h,r.typeInfo);if(d===null)return console.error(`Matrix multiplication failed. Line ${e.line}.`),null;const w=ft[r.typeInfo.name][0],I=ft[n.typeInfo.name][1],E=this.getTypeInfo(`mat${w}x${I}f`);return new ue(d,E)}if(n instanceof ue&&r instanceof M){const d=((w,I,E,m)=>{if(ft[I.name]===void 0||na[m.name]===void 0)return null;const S=ft[I.name][0],b=ft[I.name][1];if(S!==E.length)return null;const f=new Array(b);for(let _=0;_<b;_++){let v=0;for(let T=0;T<S;T++)v+=w[T*b+_]*E[T];f[_]=v}return f})(c,n.typeInfo,h,r.typeInfo);return d===null?(console.error(`Matrix vector multiplication failed. Line ${e.line}.`),null):new M(d,r.typeInfo)}if(n instanceof M&&r instanceof ue){const d=((w,I,E,m)=>{if(na[I.name]===void 0||ft[m.name]===void 0)return null;const S=ft[m.name][0],b=ft[m.name][1];if(b!==w.length)return null;const f=[];for(let _=0;_<S;_++){let v=0;for(let T=0;T<b;T++)v+=w[T]*E[T*S+_];f[_]=v}return f})(c,n.typeInfo,h,r.typeInfo);return d===null?(console.error(`Matrix vector multiplication failed. Line ${e.line}.`),null):new M(d,n.typeInfo)}{if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,I)=>w*h[I]);return new M(d,n.typeInfo)}}if(ee(i)){const c=o,h=i.map((d,w)=>d*c);return n instanceof ue?new ue(h,n.typeInfo):new M(h,n.typeInfo)}if(ee(o)){const c=i,h=o.map((d,w)=>c*d);return r instanceof ue?new ue(h,r.typeInfo):new M(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new L(a*l,u)}case"%":{if(ee(i)&&ee(o)){const c=i,h=o;if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,I)=>w%h[I]);return new M(d,n.typeInfo)}if(ee(i)){const c=o,h=i.map((d,w)=>d%c);return new M(h,n.typeInfo)}if(ee(o)){const c=i,h=o.map((d,w)=>c%d);return new M(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new L(a%l,u)}case"/":{if(ee(i)&&ee(o)){const c=i,h=o;if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,I)=>w/h[I]);return new M(d,n.typeInfo)}if(ee(i)){const c=o,h=i.map((d,w)=>d/c);return new M(h,n.typeInfo)}if(ee(o)){const c=i,h=o.map((d,w)=>c/d);return new M(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new L(a/l,u)}case"&":{if(ee(i)&&ee(o)){const c=i,h=o;if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,I)=>w&h[I]);return new M(d,n.typeInfo)}if(ee(i)){const c=o,h=i.map((d,w)=>d&c);return new M(h,n.typeInfo)}if(ee(o)){const c=i,h=o.map((d,w)=>c&d);return new M(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new L(a&l,u)}case"|":{if(ee(i)&&ee(o)){const c=i,h=o;if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,I)=>w|h[I]);return new M(d,n.typeInfo)}if(ee(i)){const c=o,h=i.map((d,w)=>d|c);return new M(h,n.typeInfo)}if(ee(o)){const c=i,h=o.map((d,w)=>c|d);return new M(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new L(a|l,u)}case"^":{if(ee(i)&&ee(o)){const c=i,h=o;if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,I)=>w^h[I]);return new M(d,n.typeInfo)}if(ee(i)){const c=o,h=i.map((d,w)=>d^c);return new M(h,n.typeInfo)}if(ee(o)){const c=i,h=o.map((d,w)=>c^d);return new M(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new L(a^l,u)}case"<<":{if(ee(i)&&ee(o)){const c=i,h=o;if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,I)=>w<<h[I]);return new M(d,n.typeInfo)}if(ee(i)){const c=o,h=i.map((d,w)=>d<<c);return new M(h,n.typeInfo)}if(ee(o)){const c=i,h=o.map((d,w)=>c<<d);return new M(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new L(a<<l,u)}case">>":{if(ee(i)&&ee(o)){const c=i,h=o;if(c.length!==h.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const d=c.map((w,I)=>w>>h[I]);return new M(d,n.typeInfo)}if(ee(i)){const c=o,h=i.map((d,w)=>d>>c);return new M(h,n.typeInfo)}if(ee(o)){const c=i,h=o.map((d,w)=>c>>d);return new M(h,r.typeInfo)}const a=i,l=o,u=this._maxFormatTypeInfo([n.typeInfo,r.typeInfo]);return new L(a>>l,u)}case">":if(ee(i)&&ee(o)){const a=i,l=o;if(a.length!==l.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const u=a.map((c,h)=>c>l[h]?1:0);return new M(u,n.typeInfo)}if(ee(i)){const a=o,l=i.map((u,c)=>u>a?1:0);return new M(l,n.typeInfo)}if(ee(o)){const a=i,l=o.map((u,c)=>a>u?1:0);return new M(l,r.typeInfo)}return new L(i>o?1:0,this.getTypeInfo("bool"));case"<":if(ee(i)&&ee(o)){const a=i,l=o;if(a.length!==l.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const u=a.map((c,h)=>c<l[h]?1:0);return new M(u,n.typeInfo)}if(ee(i)){const a=o,l=i.map((u,c)=>u<a?1:0);return new M(l,n.typeInfo)}if(ee(o)){const a=i,l=o.map((u,c)=>a<u?1:0);return new M(l,r.typeInfo)}return new L(i<o?1:0,this.getTypeInfo("bool"));case"==":if(ee(i)&&ee(o)){const a=i,l=o;if(a.length!==l.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const u=a.map((c,h)=>c===l[h]?1:0);return new M(u,n.typeInfo)}if(ee(i)){const a=o,l=i.map((u,c)=>u==a?1:0);return new M(l,n.typeInfo)}if(ee(o)){const a=i,l=o.map((u,c)=>a==u?1:0);return new M(l,r.typeInfo)}return new L(i===o?1:0,this.getTypeInfo("bool"));case"!=":if(ee(i)&&ee(o)){const a=i,l=o;if(a.length!==l.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const u=a.map((c,h)=>c!==l[h]?1:0);return new M(u,n.typeInfo)}if(ee(i)){const a=o,l=i.map((u,c)=>u!==a?1:0);return new M(l,n.typeInfo)}if(ee(o)){const a=i,l=o.map((u,c)=>a!==u?1:0);return new M(l,r.typeInfo)}return new L(i!==o?1:0,this.getTypeInfo("bool"));case">=":if(ee(i)&&ee(o)){const a=i,l=o;if(a.length!==l.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const u=a.map((c,h)=>c>=l[h]?1:0);return new M(u,n.typeInfo)}if(ee(i)){const a=o,l=i.map((u,c)=>u>=a?1:0);return new M(l,n.typeInfo)}if(ee(o)){const a=i,l=o.map((u,c)=>a>=u?1:0);return new M(l,r.typeInfo)}return new L(i>=o?1:0,this.getTypeInfo("bool"));case"<=":if(ee(i)&&ee(o)){const a=i,l=o;if(a.length!==l.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const u=a.map((c,h)=>c<=l[h]?1:0);return new M(u,n.typeInfo)}if(ee(i)){const a=o,l=i.map((u,c)=>u<=a?1:0);return new M(l,n.typeInfo)}if(ee(o)){const a=i,l=o.map((u,c)=>a<=u?1:0);return new M(l,r.typeInfo)}return new L(i<=o?1:0,this.getTypeInfo("bool"));case"&&":if(ee(i)&&ee(o)){const a=i,l=o;if(a.length!==l.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const u=a.map((c,h)=>c&&l[h]?1:0);return new M(u,n.typeInfo)}if(ee(i)){const a=o,l=i.map((u,c)=>u&&a?1:0);return new M(l,n.typeInfo)}if(ee(o)){const a=i,l=o.map((u,c)=>a&&u?1:0);return new M(l,r.typeInfo)}return new L(i&&o?1:0,this.getTypeInfo("bool"));case"||":if(ee(i)&&ee(o)){const a=i,l=o;if(a.length!==l.length)return console.error(`Vector length mismatch. Line ${e.line}.`),null;const u=a.map((c,h)=>c||l[h]?1:0);return new M(u,n.typeInfo)}if(ee(i)){const a=o,l=i.map((u,c)=>u||a?1:0);return new M(l,n.typeInfo)}if(ee(o)){const a=i,l=o.map((u,c)=>a||u?1:0);return new M(l,r.typeInfo)}return new L(i||o?1:0,this.getTypeInfo("bool"))}return console.error(`Unknown operator ${e.operator}. Line ${e.line}`),null}_evalCall(e,t){if(e.cachedReturnValue!==null)return e.cachedReturnValue;const n=t.clone();n.currentFunctionName=e.name;const r=t.getFunction(e.name);if(!r)return e.isBuiltin?this._callBuiltinFunction(e,n):this.getTypeInfo(e.name)?this._evalCreate(e,t):(console.error(`Unknown function "${e.name}". Line ${e.line}`),null);for(let i=0;i<r.node.args.length;++i){const o=r.node.args[i],a=this.evalExpression(e.args[i],n);n.createVariable(o.name,a,o)}return this._execStatements(r.node.body,n)}_callBuiltinFunction(e,t){switch(e.name){case"all":return this.builtins.All(e,t);case"any":return this.builtins.Any(e,t);case"select":return this.builtins.Select(e,t);case"arrayLength":return this.builtins.ArrayLength(e,t);case"abs":return this.builtins.Abs(e,t);case"acos":return this.builtins.Acos(e,t);case"acosh":return this.builtins.Acosh(e,t);case"asin":return this.builtins.Asin(e,t);case"asinh":return this.builtins.Asinh(e,t);case"atan":return this.builtins.Atan(e,t);case"atanh":return this.builtins.Atanh(e,t);case"atan2":return this.builtins.Atan2(e,t);case"ceil":return this.builtins.Ceil(e,t);case"clamp":return this.builtins.Clamp(e,t);case"cos":return this.builtins.Cos(e,t);case"cosh":return this.builtins.Cosh(e,t);case"countLeadingZeros":return this.builtins.CountLeadingZeros(e,t);case"countOneBits":return this.builtins.CountOneBits(e,t);case"countTrailingZeros":return this.builtins.CountTrailingZeros(e,t);case"cross":return this.builtins.Cross(e,t);case"degrees":return this.builtins.Degrees(e,t);case"determinant":return this.builtins.Determinant(e,t);case"distance":return this.builtins.Distance(e,t);case"dot":return this.builtins.Dot(e,t);case"dot4U8Packed":return this.builtins.Dot4U8Packed(e,t);case"dot4I8Packed":return this.builtins.Dot4I8Packed(e,t);case"exp":return this.builtins.Exp(e,t);case"exp2":return this.builtins.Exp2(e,t);case"extractBits":return this.builtins.ExtractBits(e,t);case"faceForward":return this.builtins.FaceForward(e,t);case"firstLeadingBit":return this.builtins.FirstLeadingBit(e,t);case"firstTrailingBit":return this.builtins.FirstTrailingBit(e,t);case"floor":return this.builtins.Floor(e,t);case"fma":return this.builtins.Fma(e,t);case"fract":return this.builtins.Fract(e,t);case"frexp":return this.builtins.Frexp(e,t);case"insertBits":return this.builtins.InsertBits(e,t);case"inverseSqrt":return this.builtins.InverseSqrt(e,t);case"ldexp":return this.builtins.Ldexp(e,t);case"length":return this.builtins.Length(e,t);case"log":return this.builtins.Log(e,t);case"log2":return this.builtins.Log2(e,t);case"max":return this.builtins.Max(e,t);case"min":return this.builtins.Min(e,t);case"mix":return this.builtins.Mix(e,t);case"modf":return this.builtins.Modf(e,t);case"normalize":return this.builtins.Normalize(e,t);case"pow":return this.builtins.Pow(e,t);case"quantizeToF16":return this.builtins.QuantizeToF16(e,t);case"radians":return this.builtins.Radians(e,t);case"reflect":return this.builtins.Reflect(e,t);case"refract":return this.builtins.Refract(e,t);case"reverseBits":return this.builtins.ReverseBits(e,t);case"round":return this.builtins.Round(e,t);case"saturate":return this.builtins.Saturate(e,t);case"sign":return this.builtins.Sign(e,t);case"sin":return this.builtins.Sin(e,t);case"sinh":return this.builtins.Sinh(e,t);case"smoothstep":return this.builtins.SmoothStep(e,t);case"sqrt":return this.builtins.Sqrt(e,t);case"step":return this.builtins.Step(e,t);case"tan":return this.builtins.Tan(e,t);case"tanh":return this.builtins.Tanh(e,t);case"transpose":return this.builtins.Transpose(e,t);case"trunc":return this.builtins.Trunc(e,t);case"dpdx":return this.builtins.Dpdx(e,t);case"dpdxCoarse":return this.builtins.DpdxCoarse(e,t);case"dpdxFine":return this.builtins.DpdxFine(e,t);case"dpdy":return this.builtins.Dpdy(e,t);case"dpdyCoarse":return this.builtins.DpdyCoarse(e,t);case"dpdyFine":return this.builtins.DpdyFine(e,t);case"fwidth":return this.builtins.Fwidth(e,t);case"fwidthCoarse":return this.builtins.FwidthCoarse(e,t);case"fwidthFine":return this.builtins.FwidthFine(e,t);case"textureDimensions":return this.builtins.TextureDimensions(e,t);case"textureGather":return this.builtins.TextureGather(e,t);case"textureGatherCompare":return this.builtins.TextureGatherCompare(e,t);case"textureLoad":return this.builtins.TextureLoad(e,t);case"textureNumLayers":return this.builtins.TextureNumLayers(e,t);case"textureNumLevels":return this.builtins.TextureNumLevels(e,t);case"textureNumSamples":return this.builtins.TextureNumSamples(e,t);case"textureSample":return this.builtins.TextureSample(e,t);case"textureSampleBias":return this.builtins.TextureSampleBias(e,t);case"textureSampleCompare":return this.builtins.TextureSampleCompare(e,t);case"textureSampleCompareLevel":return this.builtins.TextureSampleCompareLevel(e,t);case"textureSampleGrad":return this.builtins.TextureSampleGrad(e,t);case"textureSampleLevel":return this.builtins.TextureSampleLevel(e,t);case"textureSampleBaseClampToEdge":return this.builtins.TextureSampleBaseClampToEdge(e,t);case"textureStore":return this.builtins.TextureStore(e,t);case"atomicLoad":return this.builtins.AtomicLoad(e,t);case"atomicStore":return this.builtins.AtomicStore(e,t);case"atomicAdd":return this.builtins.AtomicAdd(e,t);case"atomicSub":return this.builtins.AtomicSub(e,t);case"atomicMax":return this.builtins.AtomicMax(e,t);case"atomicMin":return this.builtins.AtomicMin(e,t);case"atomicAnd":return this.builtins.AtomicAnd(e,t);case"atomicOr":return this.builtins.AtomicOr(e,t);case"atomicXor":return this.builtins.AtomicXor(e,t);case"atomicExchange":return this.builtins.AtomicExchange(e,t);case"atomicCompareExchangeWeak":return this.builtins.AtomicCompareExchangeWeak(e,t);case"pack4x8snorm":return this.builtins.Pack4x8snorm(e,t);case"pack4x8unorm":return this.builtins.Pack4x8unorm(e,t);case"pack4xI8":return this.builtins.Pack4xI8(e,t);case"pack4xU8":return this.builtins.Pack4xU8(e,t);case"pack4x8Clamp":return this.builtins.Pack4x8Clamp(e,t);case"pack4xU8Clamp":return this.builtins.Pack4xU8Clamp(e,t);case"pack2x16snorm":return this.builtins.Pack2x16snorm(e,t);case"pack2x16unorm":return this.builtins.Pack2x16unorm(e,t);case"pack2x16float":return this.builtins.Pack2x16float(e,t);case"unpack4x8snorm":return this.builtins.Unpack4x8snorm(e,t);case"unpack4x8unorm":return this.builtins.Unpack4x8unorm(e,t);case"unpack4xI8":return this.builtins.Unpack4xI8(e,t);case"unpack4xU8":return this.builtins.Unpack4xU8(e,t);case"unpack2x16snorm":return this.builtins.Unpack2x16snorm(e,t);case"unpack2x16unorm":return this.builtins.Unpack2x16unorm(e,t);case"unpack2x16float":return this.builtins.Unpack2x16float(e,t);case"storageBarrier":return this.builtins.StorageBarrier(e,t);case"textureBarrier":return this.builtins.TextureBarrier(e,t);case"workgroupBarrier":return this.builtins.WorkgroupBarrier(e,t);case"workgroupUniformLoad":return this.builtins.WorkgroupUniformLoad(e,t);case"subgroupAdd":return this.builtins.SubgroupAdd(e,t);case"subgroupExclusiveAdd":return this.builtins.SubgroupExclusiveAdd(e,t);case"subgroupInclusiveAdd":return this.builtins.SubgroupInclusiveAdd(e,t);case"subgroupAll":return this.builtins.SubgroupAll(e,t);case"subgroupAnd":return this.builtins.SubgroupAnd(e,t);case"subgroupAny":return this.builtins.SubgroupAny(e,t);case"subgroupBallot":return this.builtins.SubgroupBallot(e,t);case"subgroupBroadcast":return this.builtins.SubgroupBroadcast(e,t);case"subgroupBroadcastFirst":return this.builtins.SubgroupBroadcastFirst(e,t);case"subgroupElect":return this.builtins.SubgroupElect(e,t);case"subgroupMax":return this.builtins.SubgroupMax(e,t);case"subgroupMin":return this.builtins.SubgroupMin(e,t);case"subgroupMul":return this.builtins.SubgroupMul(e,t);case"subgroupExclusiveMul":return this.builtins.SubgroupExclusiveMul(e,t);case"subgroupInclusiveMul":return this.builtins.SubgroupInclusiveMul(e,t);case"subgroupOr":return this.builtins.SubgroupOr(e,t);case"subgroupShuffle":return this.builtins.SubgroupShuffle(e,t);case"subgroupShuffleDown":return this.builtins.SubgroupShuffleDown(e,t);case"subgroupShuffleUp":return this.builtins.SubgroupShuffleUp(e,t);case"subgroupShuffleXor":return this.builtins.SubgroupShuffleXor(e,t);case"subgroupXor":return this.builtins.SubgroupXor(e,t);case"quadBroadcast":return this.builtins.QuadBroadcast(e,t);case"quadSwapDiagonal":return this.builtins.QuadSwapDiagonal(e,t);case"quadSwapX":return this.builtins.QuadSwapX(e,t);case"quadSwapY":return this.builtins.QuadSwapY(e,t)}const n=t.getFunction(e.name);if(n){const r=t.clone();for(let i=0;i<n.node.args.length;++i){const o=n.node.args[i],a=this.evalExpression(e.args[i],r);r.setVariable(o.name,a,o)}return this._execStatements(n.node.body,r)}return null}_callConstructorValue(e,t){if(!e.args||e.args.length===0)return new L(0,this.getTypeInfo(e.type));const n=this.evalExpression(e.args[0],t);return n.typeInfo=this.getTypeInfo(e.type),n.getSubData(this,e.postfix,t).clone()}_callConstructorVec(e,t){const n=this.getTypeInfo(e.type),r=e.type.getTypeName(),i=na[r];if(i===void 0)return console.error(`Invalid vec constructor ${r}. Line ${e.line}`),null;const o=[];if(e instanceof We)if(e.isVector){const a=e.vectorValue;for(const l of a)o.push(l)}else o.push(e.scalarValue);else if(e.args)for(const a of e.args){const l=this.evalExpression(a,t);if(l instanceof M){const u=l.data;for(let c=0;c<u.length;++c){let h=u[c];o.push(h)}}else if(l instanceof L){let u=l.value;o.push(u)}}if(e.type instanceof U&&e.type.format===null&&(e.type.format=U.f32),o.length===0){const a=new Array(i).fill(0);return new M(a,n).getSubData(this,e.postfix,t)}if(o.length===1)for(;o.length<i;)o.push(o[0]);return o.length<i?(console.error(`Invalid vec constructor. Line ${e.line}`),null):new M(o.length>i?o.slice(0,i):o,n).getSubData(this,e.postfix,t)}_callConstructorMatrix(e,t){const n=this.getTypeInfo(e.type),r=e.type.getTypeName(),i=ft[r];if(i===void 0)return console.error(`Invalid matrix constructor ${r}. Line ${e.line}`),null;const o=[];if(e instanceof We)if(e.isVector){const a=e.vectorValue;for(const l of a)o.push(l)}else o.push(e.scalarValue);else if(e.args)for(const a of e.args){const l=this.evalExpression(a,t);l instanceof M?o.push(...l.data):l instanceof L?o.push(l.value):l instanceof ue&&o.push(...l.data)}if(n instanceof ps&&n.format===null&&(n.format=this.getTypeInfo("f32")),o.length===0){const a=new Array(i[2]).fill(0);return new ue(a,n).getSubData(this,e.postfix,t)}return o.length!==i[2]?(console.error(`Invalid matrix constructor. Line ${e.line}`),null):new ue(o,n).getSubData(this,e.postfix,t)}}at._breakObj=new qt(new Gt("BREAK",null),null),at._continueObj=new qt(new Gt("CONTINUE",null),null),at._priority=new Map([["f32",0],["f16",1],["u32",2],["i32",3],["x32",3]]);class Jk{constructor(){this.constants=new Map,this.aliases=new Map,this.structs=new Map}}class eT{constructor(){this._tokens=[],this._current=0,this._currentLine=1,this._deferArrayCountEval=[],this._currentLoop=[],this._context=new Jk,this._exec=new at,this._forwardTypeCount=0}parse(e){this._initialize(e),this._deferArrayCountEval.length=0;const t=[];for(;!this._isAtEnd();){const n=this._global_decl_or_directive();if(!n)break;t.push(n)}if(this._deferArrayCountEval.length>0){for(const n of this._deferArrayCountEval){const r=n.arrayType,i=n.countNode;if(i instanceof Mt){const o=i.name,a=this._context.constants.get(o);if(a)try{const l=a.constEvaluate(this._exec);r.count=l}catch{}}}this._deferArrayCountEval.length=0}if(this._forwardTypeCount>0)for(const n of t)n.search(r=>{r instanceof Lc||r instanceof Ii?r.type=this._forwardType(r.type):r instanceof gr?r.format=this._forwardType(r.format):r instanceof Cn||r instanceof mr||r instanceof Si?r.type=this._forwardType(r.type):r instanceof kr?r.returnType=this._forwardType(r.returnType):r instanceof Rc&&(r.type=this._forwardType(r.type))});return t}_forwardType(e){if(e instanceof Pc){const t=this._getType(e.name);if(t)return t}else e instanceof Ii?e.type=this._forwardType(e.type):e instanceof gr&&(e.format=this._forwardType(e.format));return e}_initialize(e){if(e)if(typeof e=="string"){const t=new Gk(e);this._tokens=t.scanTokens()}else this._tokens=e;else this._tokens=[];this._current=0}_updateNode(e,t){return e.line=t??this._currentLine,e}_error(e,t){return{token:e,message:t,toString:()=>`${t}`}}_isAtEnd(){return this._current>=this._tokens.length||this._peek().type==D.eof}_match(e){if(e instanceof F)return!!this._check(e)&&(this._advance(),!0);for(let t=0,n=e.length;t<n;++t){const r=e[t];if(this._check(r))return this._advance(),!0}return!1}_consume(e,t){if(this._check(e))return this._advance();throw this._error(this._peek(),`${t}. Line:${this._currentLine}`)}_check(e){if(this._isAtEnd())return!1;const t=this._peek();if(e instanceof Array){const n=t.type;let r=!1;for(const i of e){if(n===i)return!0;i===D.tokens.name&&(r=!0)}if(r){const i=D.tokens.name.rule.exec(t.lexeme);if(i&&i.index==0&&i[0]==t.lexeme)return!0}return!1}if(t.type===e)return!0;if(e===D.tokens.name){const n=D.tokens.name.rule.exec(t.lexeme);return n&&n.index==0&&n[0]==t.lexeme}return!1}_advance(){var e,t;return this._currentLine=(t=(e=this._peek())===null||e===void 0?void 0:e.line)!==null&&t!==void 0?t:-1,this._isAtEnd()||this._current++,this._previous()}_peek(){return this._tokens[this._current]}_previous(){return this._tokens[this._current-1]}_global_decl_or_directive(){for(;this._match(D.tokens.semicolon)&&!this._isAtEnd(););if(this._match(D.keywords.alias)){const t=this._type_alias();return this._consume(D.tokens.semicolon,"Expected ';'"),this._exec.reflection.updateAST([t]),t}if(this._match(D.keywords.diagnostic)){const t=this._diagnostic();return this._consume(D.tokens.semicolon,"Expected ';'"),this._exec.reflection.updateAST([t]),t}if(this._match(D.keywords.requires)){const t=this._requires_directive();return this._consume(D.tokens.semicolon,"Expected ';'"),this._exec.reflection.updateAST([t]),t}if(this._match(D.keywords.enable)){const t=this._enable_directive();return this._consume(D.tokens.semicolon,"Expected ';'"),this._exec.reflection.updateAST([t]),t}const e=this._attribute();if(this._check(D.keywords.var)){const t=this._global_variable_decl();return t!=null&&(t.attributes=e),this._consume(D.tokens.semicolon,"Expected ';'."),this._exec.reflection.updateAST([t]),t}if(this._check(D.keywords.override)){const t=this._override_variable_decl();return t!=null&&(t.attributes=e),this._consume(D.tokens.semicolon,"Expected ';'."),this._exec.reflection.updateAST([t]),t}if(this._check(D.keywords.let)){const t=this._global_let_decl();return t!=null&&(t.attributes=e),this._consume(D.tokens.semicolon,"Expected ';'."),this._exec.reflection.updateAST([t]),t}if(this._check(D.keywords.const)){const t=this._global_const_decl();return t!=null&&(t.attributes=e),this._consume(D.tokens.semicolon,"Expected ';'."),this._exec.reflection.updateAST([t]),t}if(this._check(D.keywords.struct)){const t=this._struct_decl();return t!=null&&(t.attributes=e),this._exec.reflection.updateAST([t]),t}if(this._check(D.keywords.fn)){const t=this._function_decl();return t!=null&&(t.attributes=e),this._exec.reflection.updateAST([t]),t}return null}_function_decl(){if(!this._match(D.keywords.fn))return null;const e=this._currentLine,t=this._consume(D.tokens.ident,"Expected function name.").toString();this._consume(D.tokens.paren_left,"Expected '(' for function arguments.");const n=[];if(!this._check(D.tokens.paren_right))do{if(this._check(D.tokens.paren_right))break;const a=this._attribute(),l=this._consume(D.tokens.name,"Expected argument name.").toString();this._consume(D.tokens.colon,"Expected ':' for argument type.");const u=this._attribute(),c=this._type_decl();c!=null&&(c.attributes=u,n.push(this._updateNode(new Rc(l,c,a))))}while(this._match(D.tokens.comma));this._consume(D.tokens.paren_right,"Expected ')' after function arguments.");let r=null;if(this._match(D.tokens.arrow)){const a=this._attribute();r=this._type_decl(),r!=null&&(r.attributes=a)}const i=this._compound_statement(),o=this._currentLine;return this._updateNode(new kr(t,n,r,i,e,o),e)}_compound_statement(){const e=[];for(this._consume(D.tokens.brace_left,"Expected '{' for block.");!this._check(D.tokens.brace_right);){const t=this._statement();t!==null&&e.push(t)}return this._consume(D.tokens.brace_right,"Expected '}' for block."),e}_statement(){for(;this._match(D.tokens.semicolon)&&!this._isAtEnd(););if(this._check(D.tokens.attr)&&this._attribute(),this._check(D.keywords.if))return this._if_statement();if(this._check(D.keywords.switch))return this._switch_statement();if(this._check(D.keywords.loop))return this._loop_statement();if(this._check(D.keywords.for))return this._for_statement();if(this._check(D.keywords.while))return this._while_statement();if(this._check(D.keywords.continuing))return this._continuing_statement();if(this._check(D.keywords.static_assert))return this._static_assert_statement();if(this._check(D.tokens.brace_left))return this._compound_statement();let e=null;if(this._check(D.keywords.return))e=this._return_statement();else if(this._check([D.keywords.var,D.keywords.let,D.keywords.const]))e=this._variable_statement();else if(this._match(D.keywords.discard))e=this._updateNode(new Uk);else if(this._match(D.keywords.break)){const t=this._updateNode(new Gp);if(this._currentLoop.length>0){const n=this._currentLoop[this._currentLoop.length-1];t.loopId=n.id}e=t,this._check(D.keywords.if)&&(this._advance(),t.condition=this._optional_paren_expression())}else if(this._match(D.keywords.continue)){const t=this._updateNode(new Wp);if(!(this._currentLoop.length>0))throw this._error(this._peek(),`Continue statement must be inside a loop. Line: ${t.line}`);{const n=this._currentLoop[this._currentLoop.length-1];t.loopId=n.id}e=t}else e=this._increment_decrement_statement()||this._func_call_statement()||this._assignment_statement();return e!=null&&this._consume(D.tokens.semicolon,"Expected ';' after statement."),e}_static_assert_statement(){if(!this._match(D.keywords.static_assert))return null;const e=this._currentLine,t=this._optional_paren_expression();return this._updateNode(new Lk(t),e)}_while_statement(){if(!this._match(D.keywords.while))return null;const e=this._updateNode(new Mp(null,null));return this._currentLoop.push(e),e.condition=this._optional_paren_expression(),this._check(D.tokens.attr)&&this._attribute(),e.body=this._compound_statement(),this._currentLoop.pop(),e}_continuing_statement(){const e=this._currentLoop.length>0?this._currentLoop[this._currentLoop.length-1].id:-1;if(!this._match(D.keywords.continuing))return null;const t=this._currentLine,n=this._compound_statement();return this._updateNode(new La(n,e),t)}_for_statement(){if(!this._match(D.keywords.for))return null;this._consume(D.tokens.paren_left,"Expected '('.");const e=this._updateNode(new Pp(null,null,null,null));return this._currentLoop.push(e),e.init=this._check(D.tokens.semicolon)?null:this._for_init(),this._consume(D.tokens.semicolon,"Expected ';'."),e.condition=this._check(D.tokens.semicolon)?null:this._short_circuit_or_expression(),this._consume(D.tokens.semicolon,"Expected ';'."),e.increment=this._check(D.tokens.paren_right)?null:this._for_increment(),this._consume(D.tokens.paren_right,"Expected ')'."),this._check(D.tokens.attr)&&this._attribute(),e.body=this._compound_statement(),this._currentLoop.pop(),e}_for_init(){return this._variable_statement()||this._func_call_statement()||this._assignment_statement()}_for_increment(){return this._func_call_statement()||this._increment_decrement_statement()||this._assignment_statement()}_variable_statement(){if(this._check(D.keywords.var)){const e=this._variable_decl();if(e===null)throw this._error(this._peek(),"Variable declaration expected.");let t=null;return this._match(D.tokens.equal)&&(t=this._short_circuit_or_expression()),this._updateNode(new Cn(e.name,e.type,e.storage,e.access,t),e.line)}if(this._match(D.keywords.let)){const e=this._currentLine,t=this._consume(D.tokens.name,"Expected name for let.").toString();let n=null;if(this._match(D.tokens.colon)){const i=this._attribute();n=this._type_decl(),n!=null&&(n.attributes=i)}this._consume(D.tokens.equal,"Expected '=' for let.");const r=this._short_circuit_or_expression();return this._updateNode(new mr(t,n,null,null,r),e)}if(this._match(D.keywords.const)){const e=this._currentLine,t=this._consume(D.tokens.name,"Expected name for const.").toString();let n=null;if(this._match(D.tokens.colon)){const i=this._attribute();n=this._type_decl(),n!=null&&(n.attributes=i)}this._consume(D.tokens.equal,"Expected '=' for const.");const r=this._short_circuit_or_expression();return n===null&&r instanceof We&&(n=r.type),this._updateNode(new Si(t,n,null,null,r),e)}return null}_increment_decrement_statement(){const e=this._current,t=this._unary_expression();if(t==null)return null;if(!this._check(D.increment_operators))return this._current=e,null;const n=this._consume(D.increment_operators,"Expected increment operator");return this._updateNode(new Rp(n.type===D.tokens.plus_plus?Es.increment:Es.decrement,t))}_assignment_statement(){let e=null;const t=this._currentLine;if(this._check(D.tokens.brace_right))return null;let n=this._match(D.tokens.underscore);if(n||(e=this._unary_expression()),!n&&e==null)return null;const r=this._consume(D.assignment_operators,"Expected assignment operator."),i=this._short_circuit_or_expression();return this._updateNode(new Lp(lr.parse(r.lexeme),e,i),t)}_func_call_statement(){if(!this._check(D.tokens.ident))return null;const e=this._currentLine,t=this._current,n=this._consume(D.tokens.ident,"Expected function name."),r=this._argument_expression_list();return r===null?(this._current=t,null):this._updateNode(new Gl(n.lexeme,r),e)}_loop_statement(){if(!this._match(D.keywords.loop))return null;this._check(D.tokens.attr)&&this._attribute(),this._consume(D.tokens.brace_left,"Expected '{' for loop.");const e=this._updateNode(new Bp([],null));this._currentLoop.push(e);let t=this._statement();for(;t!==null;){if(Array.isArray(t))for(let n of t)e.body.push(n);else e.body.push(t);if(t instanceof La){e.continuing=t;break}t=this._statement()}return this._currentLoop.pop(),this._consume(D.tokens.brace_right,"Expected '}' for loop."),e}_switch_statement(){if(!this._match(D.keywords.switch))return null;const e=this._updateNode(new Fp(null,[]));if(this._currentLoop.push(e),e.condition=this._optional_paren_expression(),this._check(D.tokens.attr)&&this._attribute(),this._consume(D.tokens.brace_left,"Expected '{' for switch."),e.cases=this._switch_body(),e.cases==null||e.cases.length==0)throw this._error(this._previous(),"Expected 'case' or 'default'.");return this._consume(D.tokens.brace_right,"Expected '}' for switch."),this._currentLoop.pop(),e}_switch_body(){const e=[];let t=!1;for(;this._check([D.keywords.default,D.keywords.case]);){if(this._match(D.keywords.case)){const n=this._case_selectors();for(const i of n)if(i instanceof ki){if(t)throw this._error(this._previous(),"Multiple default cases in switch statement.");t=!0;break}this._match(D.tokens.colon),this._check(D.tokens.attr)&&this._attribute(),this._consume(D.tokens.brace_left,"Exected '{' for switch case.");const r=this._case_body();this._consume(D.tokens.brace_right,"Exected '}' for switch case."),e.push(this._updateNode(new Xp(n,r)))}if(this._match(D.keywords.default)){if(t)throw this._error(this._previous(),"Multiple default cases in switch statement.");this._match(D.tokens.colon),this._check(D.tokens.attr)&&this._attribute(),this._consume(D.tokens.brace_left,"Exected '{' for switch default.");const n=this._case_body();this._consume(D.tokens.brace_right,"Exected '}' for switch default."),e.push(this._updateNode(new Yp(n)))}}return e}_case_selectors(){const e=[];for(this._match(D.keywords.default)?e.push(this._updateNode(new ki)):e.push(this._shift_expression());this._match(D.tokens.comma);)this._match(D.keywords.default)?e.push(this._updateNode(new ki)):e.push(this._shift_expression());return e}_case_body(){if(this._match(D.keywords.fallthrough))return this._consume(D.tokens.semicolon,"Expected ';'"),[];let e=this._statement();if(e==null)return[];e instanceof Array||(e=[e]);const t=this._case_body();return t.length==0?e:[...e,t[0]]}_if_statement(){if(!this._match(D.keywords.if))return null;const e=this._currentLine,t=this._optional_paren_expression();this._check(D.tokens.attr)&&this._attribute();const n=this._compound_statement();let r=[];this._match_elseif()&&(this._check(D.tokens.attr)&&this._attribute(),r=this._elseif_statement(r));let i=null;return this._match(D.keywords.else)&&(this._check(D.tokens.attr)&&this._attribute(),i=this._compound_statement()),this._updateNode(new Up(t,n,r,i),e)}_match_elseif(){return this._tokens[this._current].type===D.keywords.else&&this._tokens[this._current+1].type===D.keywords.if&&(this._advance(),this._advance(),!0)}_elseif_statement(e=[]){const t=this._optional_paren_expression(),n=this._compound_statement();return e.push(this._updateNode(new zk(t,n))),this._match_elseif()&&(this._check(D.tokens.attr)&&this._attribute(),this._elseif_statement(e)),e}_return_statement(){if(!this._match(D.keywords.return))return null;const e=this._short_circuit_or_expression();return this._updateNode(new zp(e))}_short_circuit_or_expression(){let e=this._short_circuit_and_expr();for(;this._match(D.tokens.or_or);)e=this._updateNode(new Jt(this._previous().toString(),e,this._short_circuit_and_expr()));return e}_short_circuit_and_expr(){let e=this._inclusive_or_expression();for(;this._match(D.tokens.and_and);)e=this._updateNode(new Jt(this._previous().toString(),e,this._inclusive_or_expression()));return e}_inclusive_or_expression(){let e=this._exclusive_or_expression();for(;this._match(D.tokens.or);)e=this._updateNode(new Jt(this._previous().toString(),e,this._exclusive_or_expression()));return e}_exclusive_or_expression(){let e=this._and_expression();for(;this._match(D.tokens.xor);)e=this._updateNode(new Jt(this._previous().toString(),e,this._and_expression()));return e}_and_expression(){let e=this._equality_expression();for(;this._match(D.tokens.and);)e=this._updateNode(new Jt(this._previous().toString(),e,this._equality_expression()));return e}_equality_expression(){const e=this._relational_expression();return this._match([D.tokens.equal_equal,D.tokens.not_equal])?this._updateNode(new Jt(this._previous().toString(),e,this._relational_expression())):e}_relational_expression(){let e=this._shift_expression();for(;this._match([D.tokens.less_than,D.tokens.greater_than,D.tokens.less_than_equal,D.tokens.greater_than_equal]);)e=this._updateNode(new Jt(this._previous().toString(),e,this._shift_expression()));return e}_shift_expression(){let e=this._additive_expression();for(;this._match([D.tokens.shift_left,D.tokens.shift_right]);)e=this._updateNode(new Jt(this._previous().toString(),e,this._additive_expression()));return e}_additive_expression(){let e=this._multiplicative_expression();for(;this._match([D.tokens.plus,D.tokens.minus]);)e=this._updateNode(new Jt(this._previous().toString(),e,this._multiplicative_expression()));return e}_multiplicative_expression(){let e=this._unary_expression();for(;this._match([D.tokens.star,D.tokens.forward_slash,D.tokens.modulo]);)e=this._updateNode(new Jt(this._previous().toString(),e,this._unary_expression()));return e}_unary_expression(){return this._match([D.tokens.minus,D.tokens.bang,D.tokens.tilde,D.tokens.star,D.tokens.and])?this._updateNode(new Ue(this._previous().toString(),this._unary_expression())):this._singular_expression()}_singular_expression(){const e=this._primary_expression(),t=this._postfix_expression();return t&&(e.postfix=t),e}_postfix_expression(){if(this._match(D.tokens.bracket_left)){const e=this._short_circuit_or_expression();this._consume(D.tokens.bracket_right,"Expected ']'.");const t=this._updateNode(new Ls(e)),n=this._postfix_expression();return n&&(t.postfix=n),t}if(this._match(D.tokens.period)){const e=this._consume(D.tokens.name,"Expected member name."),t=this._postfix_expression(),n=this._updateNode(new ms(e.lexeme));return t&&(n.postfix=t),n}return null}_getStruct(e){return this._context.aliases.has(e)?this._context.aliases.get(e).type:this._context.structs.has(e)?this._context.structs.get(e):null}_getType(e){const t=this._getStruct(e);if(t!==null)return t;switch(e){case"void":return K.void;case"bool":return K.bool;case"i32":return K.i32;case"u32":return K.u32;case"f32":return K.f32;case"f16":return K.f16;case"vec2f":return U.vec2f;case"vec3f":return U.vec3f;case"vec4f":return U.vec4f;case"vec2i":return U.vec2i;case"vec3i":return U.vec3i;case"vec4i":return U.vec4i;case"vec2u":return U.vec2u;case"vec3u":return U.vec3u;case"vec4u":return U.vec4u;case"vec2h":return U.vec2h;case"vec3h":return U.vec3h;case"vec4h":return U.vec4h;case"mat2x2f":return U.mat2x2f;case"mat2x3f":return U.mat2x3f;case"mat2x4f":return U.mat2x4f;case"mat3x2f":return U.mat3x2f;case"mat3x3f":return U.mat3x3f;case"mat3x4f":return U.mat3x4f;case"mat4x2f":return U.mat4x2f;case"mat4x3f":return U.mat4x3f;case"mat4x4f":return U.mat4x4f;case"mat2x2h":return U.mat2x2h;case"mat2x3h":return U.mat2x3h;case"mat2x4h":return U.mat2x4h;case"mat3x2h":return U.mat3x2h;case"mat3x3h":return U.mat3x3h;case"mat3x4h":return U.mat3x4h;case"mat4x2h":return U.mat4x2h;case"mat4x3h":return U.mat4x3h;case"mat4x4h":return U.mat4x4h;case"mat2x2i":return U.mat2x2i;case"mat2x3i":return U.mat2x3i;case"mat2x4i":return U.mat2x4i;case"mat3x2i":return U.mat3x2i;case"mat3x3i":return U.mat3x3i;case"mat3x4i":return U.mat3x4i;case"mat4x2i":return U.mat4x2i;case"mat4x3i":return U.mat4x3i;case"mat4x4i":return U.mat4x4i;case"mat2x2u":return U.mat2x2u;case"mat2x3u":return U.mat2x3u;case"mat2x4u":return U.mat2x4u;case"mat3x2u":return U.mat3x2u;case"mat3x3u":return U.mat3x3u;case"mat3x4u":return U.mat3x4u;case"mat4x2u":return U.mat4x2u;case"mat4x3u":return U.mat4x3u;case"mat4x4u":return U.mat4x4u}return null}_validateTypeRange(e,t){if(t.name==="i32"){if(e<-2147483648||e>2147483647)throw this._error(this._previous(),`Value out of range for i32: ${e}. Line: ${this._currentLine}.`)}else if(t.name==="u32"&&(e<0||e>4294967295))throw this._error(this._previous(),`Value out of range for u32: ${e}. Line: ${this._currentLine}.`)}_primary_expression(){if(this._match(D.tokens.ident)){const n=this._previous().toString();if(this._check(D.tokens.paren_left)){const r=this._argument_expression_list(),i=this._getType(n);return i!==null?this._updateNode(new fn(i,r)):this._updateNode(new ql(n,r))}if(this._context.constants.has(n)){const r=this._context.constants.get(n);return this._updateNode(new qp(n,r.value))}return this._updateNode(new Mt(n))}if(this._match(D.tokens.int_literal)){const n=this._previous().toString();let r=n.endsWith("i")||n.endsWith("i")?K.i32:n.endsWith("u")||n.endsWith("U")?K.u32:K.x32;const i=parseInt(n);return this._validateTypeRange(i,r),this._updateNode(new We(new L(i,this._exec.getTypeInfo(r)),r))}if(this._match(D.tokens.uint_literal)){const n=parseInt(this._previous().toString());return this._validateTypeRange(n,K.u32),this._updateNode(new We(new L(n,this._exec.getTypeInfo(K.u32)),K.u32))}if(this._match([D.tokens.decimal_float_literal,D.tokens.hex_float_literal])){let n=this._previous().toString(),r=n.endsWith("h");r&&(n=n.substring(0,n.length-1));const i=parseFloat(n);this._validateTypeRange(i,r?K.f16:K.f32);const o=r?K.f16:K.f32;return this._updateNode(new We(new L(i,this._exec.getTypeInfo(o)),o))}if(this._match([D.keywords.true,D.keywords.false])){let n=this._previous().toString()===D.keywords.true.rule;return this._updateNode(new We(new L(n?1:0,this._exec.getTypeInfo(K.bool)),K.bool))}if(this._check(D.tokens.paren_left))return this._paren_expression();if(this._match(D.keywords.bitcast)){this._consume(D.tokens.less_than,"Expected '<'.");const n=this._type_decl();this._consume(D.tokens.greater_than,"Expected '>'.");const r=this._paren_expression();return this._updateNode(new Hp(n,r))}const e=this._type_decl(),t=this._argument_expression_list();return this._updateNode(new fn(e,t))}_argument_expression_list(){if(!this._match(D.tokens.paren_left))return null;const e=[];do{if(this._check(D.tokens.paren_right))break;const t=this._short_circuit_or_expression();e.push(t)}while(this._match(D.tokens.comma));return this._consume(D.tokens.paren_right,"Expected ')' for agument list"),e}_optional_paren_expression(){this._match(D.tokens.paren_left);const e=this._short_circuit_or_expression();return this._match(D.tokens.paren_right),e}_paren_expression(){this._consume(D.tokens.paren_left,"Expected '('.");const e=this._short_circuit_or_expression();return this._consume(D.tokens.paren_right,"Expected ')'."),e}_struct_decl(){if(!this._match(D.keywords.struct))return null;const e=this._currentLine,t=this._consume(D.tokens.ident,"Expected name for struct.").toString();this._consume(D.tokens.brace_left,"Expected '{' for struct body.");const n=[];for(;!this._check(D.tokens.brace_right);){const o=this._attribute(),a=this._consume(D.tokens.name,"Expected variable name.").toString();this._consume(D.tokens.colon,"Expected ':' for struct member type.");const l=this._attribute(),u=this._type_decl();u!=null&&(u.attributes=l),this._check(D.tokens.brace_right)?this._match(D.tokens.comma):this._consume(D.tokens.comma,"Expected ',' for struct member."),n.push(this._updateNode(new Lc(a,u,o)))}this._consume(D.tokens.brace_right,"Expected '}' after struct body.");const r=this._currentLine,i=this._updateNode(new kn(t,n,e,r),e);return this._context.structs.set(t,i),i}_global_variable_decl(){const e=this._variable_decl();if(!e)return null;if(this._match(D.tokens.equal)){const t=this._const_expression();e.value=t}if(e.type!==null&&e.value instanceof We){if(e.value.type.name!=="x32"&&e.type.getTypeName()!==e.value.type.getTypeName())throw this._error(this._peek(),`Invalid cast from ${e.value.type.name} to ${e.type.name}. Line:${this._currentLine}`);e.value.isScalar&&this._validateTypeRange(e.value.scalarValue,e.type),e.value.type=e.type}else e.type===null&&e.value instanceof We&&(e.type=e.value.type.name==="x32"?K.i32:e.value.type,e.value.isScalar&&this._validateTypeRange(e.value.scalarValue,e.type));return e}_override_variable_decl(){const e=this._override_decl();return e&&this._match(D.tokens.equal)&&(e.value=this._const_expression()),e}_global_const_decl(){var e;if(!this._match(D.keywords.const))return null;const t=this._consume(D.tokens.name,"Expected variable name"),n=this._currentLine;let r=null;if(this._match(D.tokens.colon)){const l=this._attribute();r=this._type_decl(),r!=null&&(r.attributes=l)}let i=null;this._consume(D.tokens.equal,"const declarations require an assignment");const o=this._short_circuit_or_expression();try{let l=[K.f32],u=o.constEvaluate(this._exec,l);u instanceof L&&this._validateTypeRange(u.value,l[0]),l[0]instanceof U&&l[0].format===null&&u.typeInfo instanceof ps&&u.typeInfo.format!==null&&(u.typeInfo.format.name==="f16"?l[0].format=K.f16:u.typeInfo.format.name==="f32"?l[0].format=K.f32:u.typeInfo.format.name==="i32"?l[0].format=K.i32:u.typeInfo.format.name==="u32"?l[0].format=K.u32:u.typeInfo.format.name==="bool"?l[0].format=K.bool:console.error(`TODO: impelement template format type ${u.typeInfo.format.name}`)),i=this._updateNode(new We(u,l[0])),this._exec.context.setVariable(t.toString(),u)}catch{i=o}if(r!==null&&i instanceof We){if(i.type.name!=="x32"&&r.getTypeName()!==i.type.getTypeName())throw this._error(this._peek(),`Invalid cast from ${i.type.name} to ${r.name}. Line:${this._currentLine}`);i.type=r,i.isScalar&&this._validateTypeRange(i.scalarValue,i.type)}else r===null&&i instanceof We&&(r=(e=i?.type)!==null&&e!==void 0?e:K.f32,r===K.x32&&(r=K.i32));const a=this._updateNode(new Si(t.toString(),r,"","",i),n);return this._context.constants.set(a.name,a),a}_global_let_decl(){if(!this._match(D.keywords.let))return null;const e=this._currentLine,t=this._consume(D.tokens.name,"Expected variable name");let n=null;if(this._match(D.tokens.colon)){const i=this._attribute();n=this._type_decl(),n!=null&&(n.attributes=i)}let r=null;if(this._match(D.tokens.equal)&&(r=this._const_expression()),n!==null&&r instanceof We){if(r.type.name!=="x32"&&n.getTypeName()!==r.type.getTypeName())throw this._error(this._peek(),`Invalid cast from ${r.type.name} to ${n.name}. Line:${this._currentLine}`);r.type=n}else n===null&&r instanceof We&&(n=r.type.name==="x32"?K.i32:r.type);return r instanceof We&&r.isScalar&&this._validateTypeRange(r.scalarValue,n),this._updateNode(new mr(t.toString(),n,"","",r),e)}_const_expression(){return this._short_circuit_or_expression()}_variable_decl(){if(!this._match(D.keywords.var))return null;const e=this._currentLine;let t="",n="";this._match(D.tokens.less_than)&&(t=this._consume(D.storage_class,"Expected storage_class.").toString(),this._match(D.tokens.comma)&&(n=this._consume(D.access_mode,"Expected access_mode.").toString()),this._consume(D.tokens.greater_than,"Expected '>'."));const r=this._consume(D.tokens.name,"Expected variable name");let i=null;if(this._match(D.tokens.colon)){const o=this._attribute();i=this._type_decl(),i!=null&&(i.attributes=o)}return this._updateNode(new Cn(r.toString(),i,t,n,null),e)}_override_decl(){if(!this._match(D.keywords.override))return null;const e=this._consume(D.tokens.name,"Expected variable name");let t=null;if(this._match(D.tokens.colon)){const n=this._attribute();t=this._type_decl(),t!=null&&(t.attributes=n)}return this._updateNode(new Vl(e.toString(),t,null))}_diagnostic(){this._consume(D.tokens.paren_left,"Expected '('");const e=this._consume(D.tokens.ident,"Expected severity control name.");this._consume(D.tokens.comma,"Expected ','");let t=this._consume(D.tokens.ident,"Expected diagnostic rule name.").toString();return this._match(D.tokens.period)&&(t+=`.${this._consume(D.tokens.ident,"Expected diagnostic message.").toString()}`),this._consume(D.tokens.paren_right,"Expected ')'"),this._updateNode(new Vp(e.toString(),t))}_enable_directive(){const e=this._consume(D.tokens.ident,"identity expected.");return this._updateNode(new Bk(e.toString()))}_requires_directive(){const e=[this._consume(D.tokens.ident,"identity expected.").toString()];for(;this._match(D.tokens.comma);){const t=this._consume(D.tokens.ident,"identity expected.");e.push(t.toString())}return this._updateNode(new Fk(e))}_type_alias(){const e=this._consume(D.tokens.ident,"identity expected.");this._consume(D.tokens.equal,"Expected '=' for type alias.");let t=this._type_decl();if(t===null)throw this._error(this._peek(),"Expected Type for Alias.");this._context.aliases.has(t.name)&&(t=this._context.aliases.get(t.name).type);const n=this._updateNode(new Wl(e.toString(),t));return this._context.aliases.set(n.name,n),n}_type_decl(){if(this._check([D.tokens.ident,...D.texel_format,D.keywords.bool,D.keywords.f32,D.keywords.i32,D.keywords.u32])){const n=this._advance().toString();if(this._context.structs.has(n))return this._context.structs.get(n);if(this._context.aliases.has(n))return this._context.aliases.get(n).type;if(!this._getType(n)){const r=this._updateNode(new Pc(n));return this._forwardTypeCount++,r}return this._updateNode(new K(n))}let e=this._texture_sampler_types();if(e)return e;if(this._check(D.template_types)){let n=this._advance().toString(),r=null,i=null;return this._match(D.tokens.less_than)&&(r=this._type_decl(),i=null,this._match(D.tokens.comma)&&(i=this._consume(D.access_mode,"Expected access_mode for pointer").toString()),this._consume(D.tokens.greater_than,"Expected '>' for type.")),this._updateNode(new U(n,r,i))}if(this._match(D.keywords.ptr)){let n=this._previous().toString();this._consume(D.tokens.less_than,"Expected '<' for pointer.");const r=this._consume(D.storage_class,"Expected storage_class for pointer");this._consume(D.tokens.comma,"Expected ',' for pointer.");const i=this._type_decl();let o=null;return this._match(D.tokens.comma)&&(o=this._consume(D.access_mode,"Expected access_mode for pointer").toString()),this._consume(D.tokens.greater_than,"Expected '>' for pointer."),this._updateNode(new Ii(n,r.toString(),i,o))}const t=this._attribute();if(this._match(D.keywords.array)){let n=null,r=-1;const i=this._previous();let o=null;if(this._match(D.tokens.less_than)){n=this._type_decl(),this._context.aliases.has(n.name)&&(n=this._context.aliases.get(n.name).type);let l="";if(this._match(D.tokens.comma)){o=this._shift_expression();try{l=o.constEvaluate(this._exec).toString(),o=null}catch{l="1"}}this._consume(D.tokens.greater_than,"Expected '>' for array."),r=l?parseInt(l):0}const a=this._updateNode(new gr(i.toString(),t,n,r));return o&&this._deferArrayCountEval.push({arrayType:a,countNode:o}),a}return null}_texture_sampler_types(){if(this._match(D.sampler_type))return this._updateNode(new ur(this._previous().toString(),null,null));if(this._match(D.depth_texture_type))return this._updateNode(new ur(this._previous().toString(),null,null));if(this._match(D.sampled_texture_type)||this._match(D.multisampled_texture_type)){const e=this._previous();this._consume(D.tokens.less_than,"Expected '<' for sampler type.");const t=this._type_decl();return this._consume(D.tokens.greater_than,"Expected '>' for sampler type."),this._updateNode(new ur(e.toString(),t,null))}if(this._match(D.storage_texture_type)){const e=this._previous();this._consume(D.tokens.less_than,"Expected '<' for sampler type.");const t=this._consume(D.texel_format,"Invalid texel format.").toString();this._consume(D.tokens.comma,"Expected ',' after texel format.");const n=this._consume(D.access_mode,"Expected access mode for storage texture type.").toString();return this._consume(D.tokens.greater_than,"Expected '>' for sampler type."),this._updateNode(new ur(e.toString(),t,n))}return null}_attribute(){let e=[];for(;this._match(D.tokens.attr);){const t=this._consume(D.attribute_name,"Expected attribute name"),n=this._updateNode(new Qp(t.toString(),null));if(this._match(D.tokens.paren_left)){if(n.value=this._consume(D.literal_or_ident,"Expected attribute value").toString(),this._check(D.tokens.comma)){this._advance();do{const r=this._consume(D.literal_or_ident,"Expected attribute value").toString();n.value instanceof Array||(n.value=[n.value]),n.value.push(r)}while(this._match(D.tokens.comma))}this._consume(D.tokens.paren_right,"Expected ')'")}e.push(n)}return e.length==0?null:e}}class Zp extends yn{constructor(e){super(),e&&this.update(e)}update(e){const t=new eT().parse(e);this.updateAST(t)}}const gt=Pe({INDEX:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST,VERTEX:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST,STORAGE:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,UNIFORM:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST,READABLE:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST,WRITABLE:GPUBufferUsage.MAP_WRITE|GPUBufferUsage.COPY_SRC,QUERY:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),Uc={operation:"add",srcFactor:"one",dstFactor:"zero"},zc={operation:"add",srcFactor:"one",dstFactor:"one"},Vc={operation:"add",srcFactor:"one",dstFactor:"one-minus-src-alpha"},Gc={operation:"add",srcFactor:"one-minus-dst-alpha",dstFactor:"one"},Wc={operation:"add",srcFactor:"dst-alpha",dstFactor:"zero"},qc={operation:"add",srcFactor:"zero",dstFactor:"src-alpha"},Hc={operation:"add",srcFactor:"one-minus-dst-alpha",dstFactor:"zero"},jc={operation:"add",srcFactor:"zero",dstFactor:"one-minus-src-alpha"},Kc={operation:"add",srcFactor:"dst-alpha",dstFactor:"one-minus-src-alpha"},Xc={operation:"add",srcFactor:"one-minus-dst-alpha",dstFactor:"src-alpha"};Pe({COPY:Pe({color:Uc,alpha:Uc}),ADDITIVE:Pe({color:zc,alpha:zc}),SOURCE_OVER:Pe({color:Vc,alpha:Vc}),DESTINATION_OVER:Pe({color:Gc,alpha:Gc}),SOURCE_IN:Pe({color:Wc,alpha:Wc}),DESTINATION_IN:Pe({color:qc,alpha:qc}),SOURCE_OUT:Pe({color:Hc,alpha:Hc}),DESTINATION_OUT:Pe({color:jc,alpha:jc}),SOURCE_ATOP:Pe({color:Kc,alpha:Kc}),DESTINATION_ATOP:Pe({color:Xc,alpha:Xc})});var Yl="@vertex fn vertex()->@builtin(position)vec4f {return vec4f(0);}@fragment fn fragment()->@location(0)vec4f {return vec4f(0);}@compute @workgroup_size(1)fn compute(){}";let Jp=class{#n;#t;#e;Device;BindGroups=[];Reflect;#s;Pipeline;Descriptor;#i;#r;#a;#u=[];constructor(s,e,t){!s&&ie(te.DEVICE_NOT_REQUESTED),this.#n=t,this.Device=s,this.#t=e,this.#e=this.CreatePipelineLabel("Command Encoder")}CreatePipelineLabel(s){return this.#t&&s&&`${this.#t} ${s}`||""}CreatePipelineLayout(s,e){e??=this.CreatePipelineLabel(`${this.#n} Pipeline Layout`);const t=Array.isArray(s)&&s||[s];return this.Device.createPipelineLayout({label:e,bindGroupLayouts:t})}CreateTimestampWrites(s,e,t){return{querySet:s,beginningOfPassWriteIndex:e,endOfPassWriteIndex:t}}ResolveQuerySet(s,e,t=0,n=s.count,r=0){this.GetCommandEncoder(!0).resolveQuerySet(s,t,n,e,r)}CreateShaderModule(s,e,t,n){s||(s=Yl,mt(te.SHADER_CODE_NOT_FOUND)),e??=this.CreatePipelineLabel("Shader Module");const r=Array.isArray(s)&&s.join(`

`)||s;return this.Reflect=new Zp(r),this.Device.createShaderModule({label:e,code:r,sourceMap:t,compilationHints:n})}GetShaderModule(s){return s instanceof GPUShaderModule&&s||s.module}CreateBuffer(s){const e=s.label??this.CreatePipelineLabel("Buffer");return this.Device.createBuffer({label:e,...s})}CreateReadableBuffer(s){let e=typeof s=="number"&&s;e||=s.size;const t=s?.label??"Readable Buffer";return this.CreateBuffer({label:t,size:e,usage:gt.READABLE,...s})}CreateWritableBuffer(s){let e=typeof s=="number"&&s;e||=s.size;const t=s?.label??"Writable Buffer";return this.CreateBuffer({label:t,size:e,usage:gt.WRITABLE,...s})}#o(s,e,t=0,n=[]){const{format:r}=s.type,i=s.type.members??r?.members;let o=t+(s.offset??0);if(!i){const a=am((r??s.type).name),l=s.size/lm(a);return new(um(a))(e,o,l)}for(let a=0,l={},u=r?.isStruct&&s.count||1;a<u;++a)i.forEach(c=>l[c.name]=this.#o(c,e,o)),r?.isStruct&&(o+=s.stride),n.push(l);return n.length===1&&n[0]||n}CreateStorageBuffer(s,e=1){!this.Reflect&&ie(te.SHADER_MODULE_NOT_FOUND,"`CreateStorageBuffer`.\n            Use `CreateShaderModule` before creating a storage buffer.");const t=this.Reflect.storage.find(({name:u})=>s===u);!t&&ie(te.STORAGE_NOT_FOUND,`\`${s}\` in shader bindings.`);const n=typeof e=="number"&&e||e.length,r=e.label??`${s} Storage Buffer`,i=t.format.size*n,o=new ArrayBuffer(i),a=u=>(Object.keys(u).forEach(c=>{if(u[c].buffer instanceof ArrayBuffer){const h=u[c].constructor,d=i/h.BYTES_PER_ELEMENT;u[c]=new h(o,0,d)}else a(u[c])}),u),l=this.#o(t,o);return{buffer:this.CreateBuffer({label:r,size:i,usage:gt.STORAGE,...e}),[s]:l.buffer instanceof ArrayBuffer?new l.constructor(o,0,n):a(l)}}CreateUniformBuffer(s,e){!this.Reflect&&ie(te.SHADER_MODULE_NOT_FOUND,"`CreateUniformBuffer`.\n            Use `CreateShaderModule` before creating a uniform buffer.");const t=this.Reflect.uniforms.find(({name:i})=>s===i);!t&&ie(te.UNIFORM_NOT_FOUND,`\`${s}\` in shader uniforms.`),s==="resolution"&&mt(te.INVALID_UNIFORM_NAME,`\`${s}\`.`);const n=e?.label??`${s} Uniform Buffer`,r=new ArrayBuffer(t.size);return{buffer:this.CreateBuffer({label:n,size:r.byteLength,usage:gt.UNIFORM,...e}),[s]:this.#o(t,r)}}CreateUniformBufferLayout(s){!this.Reflect&&ie(te.SHADER_MODULE_NOT_FOUND,"`CreateUniformBufferLayout`.\n            Use `CreateShaderModule` before creating a uniform buffer layout.");const e=this.Reflect.uniforms.find(({name:t})=>s===t);return!e&&ie(te.UNIFORM_NOT_FOUND,`\`${s}\` in shader uniforms.`),s==="resolution"&&mt(te.INVALID_UNIFORM_NAME,`\`${s}\`.`),this.#o(e,new ArrayBuffer(e.size))}WriteBuffer(s,e,t=0,n,r){this.Device.queue.writeBuffer(s,t,e,n,r)}CopyBufferToBuffer(s,e,t=e.size,n=0,r=0){this.GetCommandEncoder(!0).copyBufferToBuffer(s,n,e,r,t)}#f(){return this.#n==="Render"&&GPUShaderStage.FRAGMENT||GPUShaderStage.COMPUTE}GetBufferMinBindingSize(s){return!this.Reflect&&ie(te.SHADER_MODULE_NOT_FOUND,"`GetBufferMinBindingSize`.\n            Use `CreateShaderModule` before requesting buffer's min binding size."),this.Reflect.getBindGroups().flat().find(({name:e})=>s===e)?.size??ie(te.BINDING_NOT_FOUND,`\`${s}\` in shader bind groups.`)}CreateBufferBindingLayout(s,e,t,n,r){return n??=this.#f(),{binding:r,visibility:n,buffer:{type:s,hasDynamicOffset:e,minBindingSize:t}}}CreateSamplerBindingLayout(s,e,t){return e??=this.#f(),{binding:t,visibility:e,sampler:{type:s}}}CreateTextureBindingLayout(s,e,t,n,r){return n??=this.#f(),{binding:r,visibility:n,texture:{sampleType:s,viewDimension:e,multisampled:t}}}CreateStorageTextureBindingLayout(s,e,t,n,r){return n??=this.#f(),{binding:r,visibility:n,storageTexture:{access:e,format:s,viewDimension:t}}}CreateExternalTextureBindingLayout(s,e){return s??=this.#f(),{binding:e,visibility:s,externalTexture:{}}}CreateBindGroupEntries(s,e=0){return Array.isArray(s)&&s.map((t,n)=>({binding:e?.[n]??n,resource:t}))||[{binding:e,resource:s}]}CreateBindGroupLayout(s,e){return e??=this.CreatePipelineLabel("Bind Group Layout"),s=Array.isArray(s)&&s.map((t,n)=>({...t,binding:t.binding??n}))||[{...s,binding:s.binding??0}],this.Device.createBindGroupLayout({entries:s,label:e})}CreateBindGroup(s,e=0,t){return t??=this.CreatePipelineLabel("Bind Group"),typeof e=="number"&&(e=this.Pipeline?this.Pipeline.getBindGroupLayout(e):ie(te.PIPELINE_NOT_FOUND,`${this.#n}Pipeline.`)),this.Device.createBindGroup({entries:s,label:t,layout:e})}SetBindGroups(s,e){const t=Array.isArray(s),n=Array.isArray(e);e=(e=t&&n?e.map(r=>Array.isArray(r)&&r||[r]):n&&e||e&&[e])&&e||[],this.BindGroups=t&&s.map((r,i)=>({bindGroup:r,dynamicOffsets:e,active:!0}))||[{bindGroup:s,dynamicOffsets:e,active:!0}]}AddBindGroups(s,e){const t=Array.isArray(s),n=Array.isArray(e);e=(e=t&&n?e.map(i=>Array.isArray(i)&&i||[i]):n&&e||e&&[e])&&e||[];const r=this.BindGroups.push(...t&&s.map(i=>({bindGroup:i,dynamicOffsets:e,active:!0}))||[{bindGroup:s,dynamicOffsets:e,active:!0}])-1;return!t&&[r]||Array.from({length:s.length}).map((i,o)=>r-o)}SetActiveBindGroups(s){s=Array.isArray(s)&&s||[s];for(let e=this.BindGroups.length;e--;)this.BindGroups[e].active=s.includes(e)}#h(){const s=this.#u.map(({bindGroup:r})=>r),e=this.#u.map(({dynamicOffsets:r})=>r),t=e.some(r=>typeof r=="number")&&e||void 0,n=this.#u.map(({active:r},i)=>r&&i).filter(r=>typeof r=="number");this.SetBindGroups(s,t),this.SetActiveBindGroups(n)}ClearBindGroups(){this.BindGroups.splice(0)}GetBindGroupsInfo(){!this.Reflect&&ie(te.SHADER_MODULE_NOT_FOUND,"`GetBindGroupsInfo`.\n            Use `CreateShaderModule` before requesting bind groups information.");const s=this.BindGroups.length,e=new Array(s),t=this.Reflect.getBindGroups();for(let n=0;n<s;++n){const{bindGroup:{label:r},dynamicOffsets:i,active:o}=this.BindGroups[n];e[n]={label:r,active:o,dynamicOffsets:i,bindings:t[n]}}return e}CreateCommandEncoder(){return this.#s=this.Device.createCommandEncoder({label:this.#e})}SetCommandEncoder(s){this.#s=s}GetCommandEncoder(s=!1){if(!this.#s){if(s){const e=`${this.#e&&`Label: "${this.#e}".`}`;mt(te.COMMAND_ENCODER_NOT_FOUND,` ${e} Creating a new one.`)}return this.CreateCommandEncoder()}return this.#s}SubmitCommandBuffer(){this.Device.queue.submit([this.#s.finish()])}SetPipeline(s){return this.Pipeline=s}SavePipelineState(){this.#a=this.Reflect,this.#r=this.Pipeline,this.#i=this.Descriptor,this.#u=[...this.BindGroups]}ResetPipelineState(){this.ClearBindGroups()}RestorePipelineState(){this.Descriptor=this.#i,this.SetPipeline(this.#r),this.Reflect=this.#a,this.#h()}set CommandEncoderLabel(s){this.#e=s}get ProgramName(){return this.#t}Destroy(){this.ResetPipelineState(),this.#a=void 0,this.#u.splice(0),this.SetCommandEncoder(void 0)}};const Tr=Pe({RENDER:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST,STORAGE:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING}),tT=Pe({ALL:"all",STENCIL:"stencil-only",DEPTH:"depth-only"}),nT=Pe({CLAMP:"clamp-to-edge",REPEAT:"repeat",MIRROR:"mirror-repeat"}),Er=Pe({NEAREST:"nearest",LINEAR:"linear"}),sT=Pe({NEVER:"never",LESS:"less",EQUAL:"equal",LESS_EQUAL:"less-equal",GREATER:"greater",NOT_EQUAL:"not-equal",GREATER_EQUAL:"greater-equal",ALWAYS:"always"});Object.freeze(Object.defineProperty({__proto__:null,ADDRESS:nT,ASPECT:tT,COMPARE:sT,FILTER:Er,USAGE:Tr},Symbol.toStringTag,{value:"Module"}));var Ql="const QUAD=array(vec2f(-1.0,-1.0),vec2f(1.0,-1.0),vec2f(1.0,1.0),vec2f(1.0,1.0),vec2f(-1.0,1.0),vec2f(-1.0,-1.0));fn GetQuadCoord(index: u32)->vec2f{return QUAD[index];}struct VertexOutput{@builtin(position)position: vec4f,@location(0)textureCoord: vec2f};@group(0)@binding(0)var Sampler: sampler;@group(0)@binding(1)var Texture: texture_2d<f32>;@vertex fn vertex(@builtin(vertex_index)index: u32)->VertexOutput {let position=GetQuadCoord(index);let coord=(position+1)*0.5;var output: VertexOutput;output.position=vec4f(position,0.0,1.0);output.textureCoord=vec2f(coord.x,1-coord.y);return output;}@fragment fn fragment(@location(0)textureCoord: vec2f)->@location(0)vec4f {return textureSample(Texture,Sampler,textureCoord);}",rT="enable dual_source_blending;struct DSBFragmentOutput{@location(0)@blend_src(0)source: vec4f,@location(0)@blend_src(1)factor: vec4f};@fragment fn dsbTextFragment(input: TextVertexOutput)->DSBFragmentOutput {var output: DSBFragmentOutput;let coverage=GetSubpixelCoverage(input.inverseTextureSize,input.distanceDelta,input.fontUV);output.source=Font.color;output.factor=vec4f(coverage,Font.color.a);return output;}",iT="override TRIPLET_FACTOR=0.6;const THRESHOLD=20.0/256.0;const MIN_GRAD=THRESHOLD*0.1;struct font{color: vec4f,back: vec4f,subpx: f32,hint: f32};struct text{matrix: mat3x3f,textureSize: vec2f};struct TextVertexOutput{@location(0)fontUV: vec2f,@location(1)screenUV: vec2f,@location(2)distanceDelta: f32,@builtin(position)position: vec4f,@location(3)inverseTextureSize: vec2f};@group(0)@binding(0)var Sampler: sampler;@group(0)@binding(1)var<uniform>Text: text;@group(0)@binding(2)var<uniform>Font: font;@group(0)@binding(3)var Texture: texture_2d<f32>;@vertex fn textVertex(@location(0)position: vec2f,@location(1)texture: vec2f,@location(2)size: f32)->TextVertexOutput{var output: TextVertexOutput;let clipSpace=Text.matrix*vec3f(position,1);output.inverseTextureSize=1.0/Text.textureSize;output.position=vec4f(clipSpace.xy,0,1);output.distanceDelta=1.0/size;output.screenUV=clipSpace.xy;output.fontUV=texture;return output;}fn GetSubpixelCoverage(size: vec2f,distance: f32,uv: vec2f)->vec3f{let sdf=textureSample(Texture,Sampler,uv).r;let sdfX=textureSample(Texture,Sampler,uv+vec2f(size.x,0)).r;let sdfY=textureSample(Texture,Sampler,uv+vec2f(0,size.y)).r;let strokeGradient=vec2f(sdfX-sdf,sdfY-sdf);let strokeGradientLength=max(length(strokeGradient),MIN_GRAD);let gradient=strokeGradient/vec2f(strokeGradientLength);let verticalGradient=abs(gradient.y);let horizontalDelta=mix(distance*1.1,distance*0.6,verticalGradient);let resultDelta=mix(distance,horizontalDelta,Font.hint);var alpha=smoothstep(0.5-resultDelta,resultDelta+0.5,sdf);alpha=pow(alpha,Font.hint*verticalGradient*0.2+1);if(alpha<THRESHOLD){discard;}let triplet=Font.subpx*gradient.x*0.5;let z=TRIPLET_FACTOR*triplet;let top=abs(z);let max=vec3f(-z,0,z);let average=vec3f(mix(top,-top-1,alpha));return clamp(max-average,vec3f(0),vec3f(1));}@fragment fn textFragment(input: TextVertexOutput)->@location(0)vec4f {let coverage=GetSubpixelCoverage(input.inverseTextureSize,input.distanceDelta,input.fontUV);return vec4f(mix(Font.back.rgb,Font.color.rgb,coverage),Font.color.a);}",em="struct Shape{color: vec4f,matrix: mat3x3f};@group(0)@binding(0)var<uniform>resolution: vec3f;fn GetClipSpace(position: vec2f)->vec2f{let clipSpace=position/resolution.xy*2-1;return clipSpace*vec2f(1,-1);}@group(0)@binding(1)var<uniform>shape: Shape;fn GetVertexClipSpace(position: vec2f)->vec4f{let matrixPosition=shape.matrix*vec3f(position,1);let clipSpace=GetClipSpace(matrixPosition.xy);return vec4f(clipSpace,0,1);}@vertex fn shapeVertex(@location(0)position: vec2f)->@builtin(position)vec4f {return GetVertexClipSpace(position);}",tm="@fragment fn shapeFragment()->@location(0)vec4f {return shape.color;}";const oT=`${em}

${tm}`,Yc=Object.freeze(Object.defineProperty({__proto__:null,Empty:Yl,Mipmaps:Ql,Quad:"const QUAD=array(vec2f(-1.0,-1.0),vec2f(1.0,-1.0),vec2f(1.0,1.0),vec2f(1.0,1.0),vec2f(-1.0,1.0),vec2f(-1.0,-1.0));fn GetQuadCoord(index: u32)->vec2f{return QUAD[index];}",Resolution:"@group(0)@binding(0)var<uniform>resolution: vec3f;fn GetClipSpace(position: vec2f)->vec2f{let clipSpace=position/resolution.xy*2-1;return clipSpace*vec2f(1,-1);}",SDFText:iT,SDFTextDSB:rT,Shape:oT,ShapeFragment:tm,ShapeVertex:em},Symbol.toStringTag,{value:"Module"}));class Qc{#n;#t;#e;#s;#i;constructor(e,t){!e&&ie(te.DEVICE_NOT_REQUESTED),this.#e=t,this.#n=e}#r(e){return e instanceof HTMLVideoElement?[e.videoWidth,e.videoHeight]:e instanceof VideoFrame?[e.codedWidth,e.codedHeight]:[e.width,e.height]}#a(e,t){const{size:n,width:r,height:i,depthOrArrayLayers:o}=e;return!n&&!r&&ie(te.TEXTURE_SIZE_NOT_FOUND,`\`${t}\` method.`),n??{width:r,height:i,depthOrArrayLayers:o}}#u(e,t){const n=e/256;n!==(0|n)&&mt(te.INVALID_BYTES_PER_ROW,`\`${t}\` options.`)}#o(e,t,n){!this.#e&&ie(te.LEGACY_RENDER_PIPELINE_NOT_FOUND,"creating a texture with mipmaps."),this.#e.SavePipelineState(),this.#e.ResetPipelineState(),this.#i&&this.#s||(this.#i=this.#e.CreateShaderModule(Ql),this.#s=this.CreateSampler(t)),this.#e.CreatePipeline({vertex:this.#e.CreateVertexState(this.#i),fragment:this.#e.CreateFragmentState(this.#i,void 0,this.#e.CreateTargetState(e.format))});for(let r=1;r<e.mipLevelCount;++r)n(r);this.#e.SubmitCommandBuffer(),this.#e.SetCommandEncoder(void 0),this.#e.RestorePipelineState(),this.#i=this.#s=void 0}CreateSampler(e){if(!e)return this.#n.createSampler();const{addressModeUV:t,addressMode:n,minMagFilter:r,filter:i}=e;return t&&(e.addressModeU=e.addressModeV=t),n&&(e.addressModeU=e.addressModeV=e.addressModeW=n),r&&(e.minFilter=e.magFilter=r),i&&(e.minFilter=e.magFilter=e.mipmapFilter=i),this.#n.createSampler(e)}CreateTexture(e){const t=e.label??"Texture",{format:n="rgba8unorm",usage:r=Tr.RENDER}=e;return this.#n.createTexture({label:t,format:n,usage:r,...e})}WriteTexture(e,t){const{texture:n,mipLevel:r,origin:i,aspect:o,offset:a,rowsPerImage:l}=t,[u,c]=this.#r(n);let{bytesPerRow:h}=t;h??=(t.width??u)*Float32Array.BYTES_PER_ELEMENT,this.#n.queue.writeTexture({texture:n,mipLevel:r,origin:i,aspect:o},e,{offset:a,bytesPerRow:h,rowsPerImage:l},this.#a({width:u,height:c,...t},"WriteTexture"))}CreateStorageTexture(e){let{size:t}=e;const n=Tr.STORAGE|e.usage,r=e.label??"Storage Texture",{format:i=this.PreferredStorageFormat}=e;return t=this.#e&&!t?this.#e.CanvasSize:t,this.CreateTexture({label:r,size:t,format:i,...e,usage:n})}CreateBitmapImage(e,t){return createImageBitmap(e,t)}CreateTextureFromSource(e,t={}){const n=(t=typeof t=="boolean"&&{}||t).size,r=t.size,i=t.mipLevelCount??((t.mipmaps??!0)&&this.GetMipmapLevels(e)||void 0),o=Array.isArray(t.size)||!t.size?n??this.#r(e):[r.width,r.height];return this.CreateTexture({size:o,mipLevelCount:i,...t})}ImportExternalTexture(e,t,n){return this.#n.importExternalTexture({source:e,label:t,colorSpace:n})}CreateMultisampleTexture(e=!1,t=4,n){!this.#e&&ie(te.LEGACY_RENDER_PIPELINE_NOT_FOUND,"creating a multisample texture.");const{width:r,height:i,format:o}=this.#e.CurrentTexture;return!e&&this.#t&&this.#t.width===r&&this.#t.height===i||(this.#t?.destroy(),this.#t=this.CreateTexture({usage:GPUTextureUsage.RENDER_ATTACHMENT,label:n??"Multisample Texture",size:[r,i],sampleCount:t,format:o})),this.#t}CopyImageToTexture(e,t={create:!0}){let{create:n,texture:r}=t;const[i,o]=this.#r(e),{flipY:a,mipLevel:l,aspect:u,colorSpace:c,premultipliedAlpha:h,mipmaps:d}=t;return d===!1&&((n=typeof n=="object"&&n||{}).mipmaps??=!1),!r&&!n&&ie(te.TEXTURE_NOT_FOUND,"`CopyImageToTexture`."),r??=this.CreateTextureFromSource(e,n),this.#n.queue.copyExternalImageToTexture({source:e,origin:t.sourceOrigin,flipY:a},{texture:r,mipLevel:l,origin:t.destinationOrigin,aspect:u,colorSpace:c,premultipliedAlpha:h},this.#a({width:i,height:o,...t},"CopyImageToTexture")),(d??1)&&1<r.mipLevelCount&&(r.depthOrArrayLayers===1?this.GenerateMipmaps(r):this.GenerateCubeMipmaps(r)),r}CopyTextureToTexture(e){const{source:t,create:n}=e;let{srcTexture:r,dstTexture:i}=e;!this.#e&&ie(te.LEGACY_RENDER_PIPELINE_NOT_FOUND,"copying a texture to a texture."),!r&&!t&&!n&&ie(te.TEXTURE_NOT_FOUND,"`CopyTextureToTexture`."),r??=this.CreateTextureFromSource(t,n),i??=this.CreateTextureFromSource(r,n);const{srcMipLevel:o,srcOrigin:a,srcAspect:l}=e,{dstMipLevel:u,dstOrigin:c,dstAspect:h}=e,[d,w]=this.#r(r);this.#e.GetCommandEncoder(!0).copyTextureToTexture({texture:r,mipLevel:o,origin:a,aspect:l},{texture:i,mipLevel:u,origin:c,aspect:h},this.#a({width:d,height:w,...e},"CopyTextureToTexture"))}CopyTextureToBuffer(e){const{source:t,create:n}=e;let{texture:r,bytesPerRow:i}=e;!this.#e&&ie(te.LEGACY_RENDER_PIPELINE_NOT_FOUND,"copying a texture to a buffer."),!r&&!t&&!n&&ie(te.TEXTURE_NOT_FOUND,"`CopyTextureToBuffer`."),r??=this.CreateTextureFromSource(t,n);const[o,a]=this.#r(r),{buffer:l,offset:u,rowsPerImage:c,mipLevel:h,origin:d,aspect:w}=e;i??=(e.width??o)*Float32Array.BYTES_PER_ELEMENT,this.#u(i,"CopyTextureToBuffer"),this.#e.GetCommandEncoder(!0).copyTextureToBuffer({texture:r,mipLevel:h,origin:d,aspect:w},{buffer:l,offset:u,bytesPerRow:i,rowsPerImage:c},this.#a({width:o,height:a,...e},"CopyTextureToBuffer"))}CopyBufferToTexture(e){const{source:t,create:n}=e;let{texture:r,bytesPerRow:i}=e;!this.#e&&ie(te.LEGACY_RENDER_PIPELINE_NOT_FOUND,"copying a buffer to a texture."),!r&&!t&&!n&&ie(te.TEXTURE_NOT_FOUND,"`CopyBufferToTexture`."),r??=this.CreateTextureFromSource(t,n);const[o,a]=this.#r(r),{buffer:l,offset:u,rowsPerImage:c,mipLevel:h,origin:d,aspect:w}=e;i??=(e.width??o)*Float32Array.BYTES_PER_ELEMENT,this.#u(i,"CopyBufferToTexture"),this.#e.GetCommandEncoder(!0).copyBufferToTexture({buffer:l,offset:u,bytesPerRow:i,rowsPerImage:c},{texture:r,mipLevel:h,origin:d,aspect:w},this.#a({width:o,height:a,...e},"CopyBufferToTexture"))}get PreferredStorageFormat(){const e=zt.PreferredCanvasFormat;return this.#n.features.has("bgra8unorm-storage")&&e==="bgra8unorm"?e:"rgba8unorm"}GenerateCubeMipmaps(e){this.#o(e,{minMagFilter:Er.LINEAR},t=>{for(let n=0;n<e.depthOrArrayLayers;++n)this.#e.SetBindGroups(this.#e.CreateBindGroup(this.#e.CreateBindGroupEntries([this.#s,e.createView({baseMipLevel:t-1,arrayLayerCount:1,baseArrayLayer:n,mipLevelCount:1,dimension:"2d"})]))),this.#e.CreatePassDescriptor(this.#e.CreateColorAttachment(e.createView({arrayLayerCount:1,baseArrayLayer:n,mipLevelCount:1,dimension:"2d",baseMipLevel:t}))),this.#e.Render(6,!1),this.#e.DestroyCurrentPass()})}GenerateMipmaps(e){this.#o(e,{minFilter:Er.LINEAR},t=>{this.#e.SetBindGroups(this.#e.CreateBindGroup(this.#e.CreateBindGroupEntries([this.#s,e.createView({baseMipLevel:t-1,mipLevelCount:1})]))),this.#e.CreatePassDescriptor(this.#e.CreateColorAttachment(e.createView({baseMipLevel:t,mipLevelCount:1}))),this.#e.Render(6,!1),this.#e.DestroyCurrentPass()})}GetMipmapLevels(e){const[t,n]=this.#r(e);return 1+(0|Math.log2(Math.max(t,n)))}set LegacyRenderer(e){this.#e=e}SetRenderer(e){this.LegacyRenderer=e}Destroy(){this.#t=this.#t?.destroy()}}class Zl{#n;#t;#e;#s;#i;constructor(e,t){!e&&ie(te.DEVICE_NOT_REQUESTED),this.#e=t,this.#n=e}#r(e){return e instanceof HTMLVideoElement?[e.videoWidth,e.videoHeight]:e instanceof VideoFrame?[e.codedWidth,e.codedHeight]:[e.width,e.height]}#a(e,t){const{size:n,width:r,height:i,depthOrArrayLayers:o}=e;return!n&&!r&&ie(te.TEXTURE_SIZE_NOT_FOUND,`\`${t}\` method.`),n??{width:r,height:i,depthOrArrayLayers:o}}#u(e,t){const n=e/256;n!==(0|n)&&mt(te.INVALID_BYTES_PER_ROW,`\`${t}\` options.`)}CreateSampler(e){if(!e)return this.#n.createSampler();const{addressModeUV:t,addressMode:n,minMagFilter:r,filter:i}=e;return t&&(e.addressModeU=e.addressModeV=t),n&&(e.addressModeU=e.addressModeV=e.addressModeW=n),r&&(e.minFilter=e.magFilter=r),i&&(e.minFilter=e.magFilter=e.mipmapFilter=i),this.#n.createSampler(e)}CreateTexture(e){const t=e.label??"Texture",{format:n=zt.PreferredCanvasFormat,usage:r=Tr.RENDER}=e;return this.#n.createTexture({label:t,format:n,usage:r,...e})}WriteTexture(e,t){const{texture:n,mipLevel:r,origin:i,aspect:o,offset:a,rowsPerImage:l}=t,[u,c]=this.#r(n);let{bytesPerRow:h}=t;h??=(t.width??u)*Float32Array.BYTES_PER_ELEMENT,this.#n.queue.writeTexture({texture:n,mipLevel:r,origin:i,aspect:o},e,{offset:a,bytesPerRow:h,rowsPerImage:l},this.#a({width:u,height:c,...t},"WriteTexture"))}CreateStorageTexture(e){let{size:t}=e;const n=Tr.STORAGE|e.usage,r=e.label??"Storage Texture",{format:i=this.PreferredStorageFormat}=e;return t=this.#e&&!t?this.#e.CanvasSize:t,this.CreateTexture({label:r,size:t,format:i,...e,usage:n})}CreateBitmapImage(e,t){return createImageBitmap(e,t)}CreateTextureFromSource(e,t={}){const n=(t=typeof t=="boolean"&&{}||t).size,r=t.size,i=t.mipLevelCount??((t.mipmaps??!0)&&this.GetMipmapLevels(e)||void 0),o=Array.isArray(t.size)||!t.size?n??this.#r(e):[r.width,r.height];return this.CreateTexture({size:o,mipLevelCount:i,...t})}ImportExternalTexture(e,t,n){return this.#n.importExternalTexture({source:e,label:t,colorSpace:n})}async LoadExternalImageSource(e,t={}){return new Promise(n=>{const r=new Image;for(const i in t)r[i]=t[i];r.onload=()=>n(r),r.src=e})}CreateMultisampleTexture(e=!1,t=4,n){!this.#e&&ie(te.RENDERER_NOT_FOUND,"creating a multisample texture.");const{width:r,height:i,format:o}=this.#e.CurrentTexture;return!e&&this.#t&&this.#t.width===r&&this.#t.height===i||(this.#t?.destroy(),this.#t=this.CreateTexture({usage:GPUTextureUsage.RENDER_ATTACHMENT,label:n??"Multisample Texture",size:[r,i],sampleCount:t,format:o})),this.#t}async CopyImageToTexture(e,t={create:!0}){let{create:n,texture:r}=t;const[i,o]=this.#r(e),{flipY:a,mipLevel:l,aspect:u,colorSpace:c,premultipliedAlpha:h,mipmaps:d}=t;return d===!1&&((n=typeof n=="object"&&n||{}).mipmaps??=!1),!r&&!n&&ie(te.TEXTURE_NOT_FOUND,"`CopyImageToTexture`."),r??=this.CreateTextureFromSource(e,n),this.#n.queue.copyExternalImageToTexture({source:e,origin:t.sourceOrigin,flipY:a},{texture:r,mipLevel:l,origin:t.destinationOrigin,aspect:u,colorSpace:c,premultipliedAlpha:h},this.#a({width:i,height:o,...t},"CopyImageToTexture")),(d??1)&&1<r.mipLevelCount&&(r.depthOrArrayLayers===1?await this.GenerateMipmaps(r):await this.GenerateCubeMipmaps(r)),r}async#o(e,t,n){!this.#e&&ie(te.RENDERER_NOT_FOUND,"creating a texture with mipmaps.");const r=new this.#e.Pipeline;r.DestroyPassEncoder=!0,r.SetDrawParams(6),this.#i&&this.#s||(this.#i=r.CreateShaderModule(Ql),this.#s=this.CreateSampler(t)),await this.#e.AddPipeline(r,{vertex:r.CreateVertexState(this.#i),fragment:r.CreateFragmentState(this.#i,void 0,r.CreateColorTargetState(e.format))});for(let i=1;i<e.mipLevelCount;++i)n(r,i);this.#e.SubmitCommandBuffer(),this.#e.CommandEncoder=void 0,this.#e.RemovePipeline(r),this.#i=this.#s=void 0}async GenerateCubeMipmaps(e){return this.#o(e,{minMagFilter:Er.LINEAR},(t,n)=>{for(let r=0;r<e.depthOrArrayLayers;++r)t.SetBindGroups(t.CreateBindGroup(t.CreateBindGroupEntries([this.#s,e.createView({baseMipLevel:n-1,arrayLayerCount:1,baseArrayLayer:r,mipLevelCount:1,dimension:"2d"})]))),this.#e.CreatePassDescriptor(this.#e.CreateColorAttachment(void 0,e.createView({arrayLayerCount:1,baseArrayLayer:r,mipLevelCount:1,dimension:"2d",baseMipLevel:n}))),this.#e.Render(!1)})}async GenerateMipmaps(e){return this.#o(e,{minFilter:Er.LINEAR},(t,n)=>{t.SetBindGroups(t.CreateBindGroup(t.CreateBindGroupEntries([this.#s,e.createView({baseMipLevel:n-1,mipLevelCount:1})]))),this.#e.CreatePassDescriptor(this.#e.CreateColorAttachment(void 0,e.createView({baseMipLevel:n,mipLevelCount:1}))),this.#e.Render(!1)})}CopyTextureToTexture(e){const{source:t,create:n}=e;let{srcTexture:r,dstTexture:i}=e;!this.#e&&ie(te.RENDERER_NOT_FOUND,"copying a texture to a texture."),!r&&!t&&!n&&ie(te.TEXTURE_NOT_FOUND,"`CopyTextureToTexture`."),r??=this.CreateTextureFromSource(t,n),i??=this.CreateTextureFromSource(r,n);const{srcMipLevel:o,srcOrigin:a,srcAspect:l}=e,{dstMipLevel:u,dstOrigin:c,dstAspect:h}=e,[d,w]=this.#r(r);this.#e.GetCommandEncoder(!0).copyTextureToTexture({texture:r,mipLevel:o,origin:a,aspect:l},{texture:i,mipLevel:u,origin:c,aspect:h},this.#a({width:d,height:w,...e},"CopyTextureToTexture"))}CopyTextureToBuffer(e){const{source:t,create:n}=e;let{texture:r,bytesPerRow:i}=e;!this.#e&&ie(te.RENDERER_NOT_FOUND,"copying a texture to a buffer."),!r&&!t&&!n&&ie(te.TEXTURE_NOT_FOUND,"`CopyTextureToBuffer`."),r??=this.CreateTextureFromSource(t,n);const[o,a]=this.#r(r),{buffer:l,offset:u,rowsPerImage:c,mipLevel:h,origin:d,aspect:w}=e;i??=(e.width??o)*Float32Array.BYTES_PER_ELEMENT,this.#u(i,"CopyTextureToBuffer"),this.#e.GetCommandEncoder(!0).copyTextureToBuffer({texture:r,mipLevel:h,origin:d,aspect:w},{buffer:l,offset:u,bytesPerRow:i,rowsPerImage:c},this.#a({width:o,height:a,...e},"CopyTextureToBuffer"))}CopyBufferToTexture(e){const{source:t,create:n}=e;let{texture:r,bytesPerRow:i}=e;!this.#e&&ie(te.RENDERER_NOT_FOUND,"copying a buffer to a texture."),!r&&!t&&!n&&ie(te.TEXTURE_NOT_FOUND,"`CopyBufferToTexture`."),r??=this.CreateTextureFromSource(t,n);const[o,a]=this.#r(r),{buffer:l,offset:u,rowsPerImage:c,mipLevel:h,origin:d,aspect:w}=e;i??=(e.width??o)*Float32Array.BYTES_PER_ELEMENT,this.#u(i,"CopyBufferToTexture"),this.#e.GetCommandEncoder(!0).copyBufferToTexture({buffer:l,offset:u,bytesPerRow:i,rowsPerImage:c},{texture:r,mipLevel:h,origin:d,aspect:w},this.#a({width:o,height:a,...e},"CopyBufferToTexture"))}get PreferredStorageFormat(){const e=zt.PreferredCanvasFormat;return this.#n.features.has("bgra8unorm-storage")&&e==="bgra8unorm"?e:"rgba8unorm"}GetMipmapLevels(e){const[t,n]=this.#r(e);return 1+(0|Math.log2(Math.max(t,n)))}set Renderer(e){this.#e=e}Destroy(){this.#t=this.#t?.destroy()}}const aT=(Zc=Array,Jc=s=>s.fill(0),class extends Zc{constructor(...s){super(...s),Jc(this)}});var Zc,Jc;let pe=1e-6;const eh=new Map;function nm(s){let e=eh.get(s);return e||(e=(t=>{function n(p=0,y=0){const x=new t(2);return p!==void 0&&(x[0]=p,y!==void 0&&(x[1]=y)),x}function r(p,y,x){const k=x??new t(2);return k[0]=p[0]-y[0],k[1]=p[1]-y[1],k}const i=r;function o(p,y,x,k){const C=k??new t(2);return C[0]=p[0]+x*(y[0]-p[0]),C[1]=p[1]+x*(y[1]-p[1]),C}function a(p,y,x){const k=x??new t(2);return k[0]=p[0]*y,k[1]=p[1]*y,k}const l=a;function u(p,y){const x=y??new t(2);return x[0]=1/p[0],x[1]=1/p[1],x}const c=u;function h(p,y){return p[0]*y[0]+p[1]*y[1]}function d(p){const y=p[0],x=p[1];return Math.sqrt(y*y+x*x)}const w=d;function I(p){const y=p[0],x=p[1];return y*y+x*x}const E=I;function m(p,y){const x=p[0]-y[0],k=p[1]-y[1];return Math.sqrt(x*x+k*k)}const S=m;function b(p,y){const x=p[0]-y[0],k=p[1]-y[1];return x*x+k*k}const f=b;function _(p,y){const x=y??new t(2),k=p[0],C=p[1],R=Math.sqrt(k*k+C*C);return R>1e-5?(x[0]=k/R,x[1]=C/R):(x[0]=0,x[1]=0),x}function v(p,y){const x=y??new t(2);return x[0]=p[0],x[1]=p[1],x}const T=v;function N(p,y,x){const k=x??new t(2);return k[0]=p[0]*y[0],k[1]=p[1]*y[1],k}const O=N;function $(p,y,x){const k=x??new t(2);return k[0]=p[0]/y[0],k[1]=p[1]/y[1],k}const A=$;function g(p,y,x){const k=x??new t(2);return _(p,k),a(k,y,k)}return{create:n,fromValues:n,set(p,y,x){const k=x??new t(2);return k[0]=p,k[1]=y,k},ceil(p,y){const x=y??new t(2);return x[0]=Math.ceil(p[0]),x[1]=Math.ceil(p[1]),x},floor(p,y){const x=y??new t(2);return x[0]=Math.floor(p[0]),x[1]=Math.floor(p[1]),x},round(p,y){const x=y??new t(2);return x[0]=Math.round(p[0]),x[1]=Math.round(p[1]),x},clamp(p,y=0,x=1,k){const C=k??new t(2);return C[0]=Math.min(x,Math.max(y,p[0])),C[1]=Math.min(x,Math.max(y,p[1])),C},add(p,y,x){const k=x??new t(2);return k[0]=p[0]+y[0],k[1]=p[1]+y[1],k},addScaled(p,y,x,k){const C=k??new t(2);return C[0]=p[0]+y[0]*x,C[1]=p[1]+y[1]*x,C},angle(p,y){const x=p[0],k=p[1],C=y[0],R=y[1],z=Math.sqrt(x*x+k*k)*Math.sqrt(C*C+R*R),j=z&&h(p,y)/z;return Math.acos(j)},subtract:r,sub:i,equalsApproximately(p,y){return Math.abs(p[0]-y[0])<pe&&Math.abs(p[1]-y[1])<pe},equals(p,y){return p[0]===y[0]&&p[1]===y[1]},lerp:o,lerpV(p,y,x,k){const C=k??new t(2);return C[0]=p[0]+x[0]*(y[0]-p[0]),C[1]=p[1]+x[1]*(y[1]-p[1]),C},max(p,y,x){const k=x??new t(2);return k[0]=Math.max(p[0],y[0]),k[1]=Math.max(p[1],y[1]),k},min(p,y,x){const k=x??new t(2);return k[0]=Math.min(p[0],y[0]),k[1]=Math.min(p[1],y[1]),k},mulScalar:a,scale:l,divScalar(p,y,x){const k=x??new t(2);return k[0]=p[0]/y,k[1]=p[1]/y,k},inverse:u,invert:c,cross(p,y,x){const k=x??new t(3),C=p[0]*y[1]-p[1]*y[0];return k[0]=0,k[1]=0,k[2]=C,k},dot:h,length:d,len:w,lengthSq:I,lenSq:E,distance:m,dist:S,distanceSq:b,distSq:f,normalize:_,negate(p,y){const x=y??new t(2);return x[0]=-p[0],x[1]=-p[1],x},copy:v,clone:T,multiply:N,mul:O,divide:$,div:A,random(p=1,y){const x=y??new t(2),k=2*Math.random()*Math.PI;return x[0]=Math.cos(k)*p,x[1]=Math.sin(k)*p,x},zero(p){const y=p??new t(2);return y[0]=0,y[1]=0,y},transformMat4(p,y,x){const k=x??new t(2),C=p[0],R=p[1];return k[0]=C*y[0]+R*y[4]+y[12],k[1]=C*y[1]+R*y[5]+y[13],k},transformMat3(p,y,x){const k=x??new t(2),C=p[0],R=p[1];return k[0]=y[0]*C+y[4]*R+y[8],k[1]=y[1]*C+y[5]*R+y[9],k},rotate(p,y,x,k){const C=k??new t(2),R=p[0]-y[0],z=p[1]-y[1],j=Math.sin(x),G=Math.cos(x);return C[0]=R*G-z*j+y[0],C[1]=R*j+z*G+y[1],C},setLength:g,truncate(p,y,x){const k=x??new t(2);return d(p)>y?g(p,y,k):v(p,k)},midpoint(p,y,x){return o(p,y,.5,x??new t(2))}}})(s),eh.set(s,e)),e}const th=new Map;function $o(s){let e=th.get(s);return e||(e=(t=>{function n(p,y,x){const k=new t(3);return p!==void 0&&(k[0]=p,y!==void 0&&(k[1]=y,x!==void 0&&(k[2]=x))),k}function r(p,y,x){const k=x??new t(3);return k[0]=p[0]-y[0],k[1]=p[1]-y[1],k[2]=p[2]-y[2],k}const i=r;function o(p,y,x,k){const C=k??new t(3);return C[0]=p[0]+x*(y[0]-p[0]),C[1]=p[1]+x*(y[1]-p[1]),C[2]=p[2]+x*(y[2]-p[2]),C}function a(p,y,x){const k=x??new t(3);return k[0]=p[0]*y,k[1]=p[1]*y,k[2]=p[2]*y,k}const l=a;function u(p,y){const x=y??new t(3);return x[0]=1/p[0],x[1]=1/p[1],x[2]=1/p[2],x}const c=u;function h(p,y){return p[0]*y[0]+p[1]*y[1]+p[2]*y[2]}function d(p){const y=p[0],x=p[1],k=p[2];return Math.sqrt(y*y+x*x+k*k)}const w=d;function I(p){const y=p[0],x=p[1],k=p[2];return y*y+x*x+k*k}const E=I;function m(p,y){const x=p[0]-y[0],k=p[1]-y[1],C=p[2]-y[2];return Math.sqrt(x*x+k*k+C*C)}const S=m;function b(p,y){const x=p[0]-y[0],k=p[1]-y[1],C=p[2]-y[2];return x*x+k*k+C*C}const f=b;function _(p,y){const x=y??new t(3),k=p[0],C=p[1],R=p[2],z=Math.sqrt(k*k+C*C+R*R);return z>1e-5?(x[0]=k/z,x[1]=C/z,x[2]=R/z):(x[0]=0,x[1]=0,x[2]=0),x}function v(p,y){const x=y??new t(3);return x[0]=p[0],x[1]=p[1],x[2]=p[2],x}const T=v;function N(p,y,x){const k=x??new t(3);return k[0]=p[0]*y[0],k[1]=p[1]*y[1],k[2]=p[2]*y[2],k}const O=N;function $(p,y,x){const k=x??new t(3);return k[0]=p[0]/y[0],k[1]=p[1]/y[1],k[2]=p[2]/y[2],k}const A=$;function g(p,y,x){const k=x??new t(3);return _(p,k),a(k,y,k)}return{create:n,fromValues:n,set(p,y,x,k){const C=k??new t(3);return C[0]=p,C[1]=y,C[2]=x,C},ceil(p,y){const x=y??new t(3);return x[0]=Math.ceil(p[0]),x[1]=Math.ceil(p[1]),x[2]=Math.ceil(p[2]),x},floor(p,y){const x=y??new t(3);return x[0]=Math.floor(p[0]),x[1]=Math.floor(p[1]),x[2]=Math.floor(p[2]),x},round(p,y){const x=y??new t(3);return x[0]=Math.round(p[0]),x[1]=Math.round(p[1]),x[2]=Math.round(p[2]),x},clamp(p,y=0,x=1,k){const C=k??new t(3);return C[0]=Math.min(x,Math.max(y,p[0])),C[1]=Math.min(x,Math.max(y,p[1])),C[2]=Math.min(x,Math.max(y,p[2])),C},add(p,y,x){const k=x??new t(3);return k[0]=p[0]+y[0],k[1]=p[1]+y[1],k[2]=p[2]+y[2],k},addScaled(p,y,x,k){const C=k??new t(3);return C[0]=p[0]+y[0]*x,C[1]=p[1]+y[1]*x,C[2]=p[2]+y[2]*x,C},angle(p,y){const x=p[0],k=p[1],C=p[2],R=y[0],z=y[1],j=y[2],G=Math.sqrt(x*x+k*k+C*C)*Math.sqrt(R*R+z*z+j*j),X=G&&h(p,y)/G;return Math.acos(X)},subtract:r,sub:i,equalsApproximately(p,y){return Math.abs(p[0]-y[0])<pe&&Math.abs(p[1]-y[1])<pe&&Math.abs(p[2]-y[2])<pe},equals(p,y){return p[0]===y[0]&&p[1]===y[1]&&p[2]===y[2]},lerp:o,lerpV(p,y,x,k){const C=k??new t(3);return C[0]=p[0]+x[0]*(y[0]-p[0]),C[1]=p[1]+x[1]*(y[1]-p[1]),C[2]=p[2]+x[2]*(y[2]-p[2]),C},max(p,y,x){const k=x??new t(3);return k[0]=Math.max(p[0],y[0]),k[1]=Math.max(p[1],y[1]),k[2]=Math.max(p[2],y[2]),k},min(p,y,x){const k=x??new t(3);return k[0]=Math.min(p[0],y[0]),k[1]=Math.min(p[1],y[1]),k[2]=Math.min(p[2],y[2]),k},mulScalar:a,scale:l,divScalar(p,y,x){const k=x??new t(3);return k[0]=p[0]/y,k[1]=p[1]/y,k[2]=p[2]/y,k},inverse:u,invert:c,cross(p,y,x){const k=x??new t(3),C=p[2]*y[0]-p[0]*y[2],R=p[0]*y[1]-p[1]*y[0];return k[0]=p[1]*y[2]-p[2]*y[1],k[1]=C,k[2]=R,k},dot:h,length:d,len:w,lengthSq:I,lenSq:E,distance:m,dist:S,distanceSq:b,distSq:f,normalize:_,negate(p,y){const x=y??new t(3);return x[0]=-p[0],x[1]=-p[1],x[2]=-p[2],x},copy:v,clone:T,multiply:N,mul:O,divide:$,div:A,random(p=1,y){const x=y??new t(3),k=2*Math.random()*Math.PI,C=2*Math.random()-1,R=Math.sqrt(1-C*C)*p;return x[0]=Math.cos(k)*R,x[1]=Math.sin(k)*R,x[2]=C*p,x},zero(p){const y=p??new t(3);return y[0]=0,y[1]=0,y[2]=0,y},transformMat4(p,y,x){const k=x??new t(3),C=p[0],R=p[1],z=p[2],j=y[3]*C+y[7]*R+y[11]*z+y[15]||1;return k[0]=(y[0]*C+y[4]*R+y[8]*z+y[12])/j,k[1]=(y[1]*C+y[5]*R+y[9]*z+y[13])/j,k[2]=(y[2]*C+y[6]*R+y[10]*z+y[14])/j,k},transformMat4Upper3x3(p,y,x){const k=x??new t(3),C=p[0],R=p[1],z=p[2];return k[0]=C*y[0]+R*y[4]+z*y[8],k[1]=C*y[1]+R*y[5]+z*y[9],k[2]=C*y[2]+R*y[6]+z*y[10],k},transformMat3(p,y,x){const k=x??new t(3),C=p[0],R=p[1],z=p[2];return k[0]=C*y[0]+R*y[4]+z*y[8],k[1]=C*y[1]+R*y[5]+z*y[9],k[2]=C*y[2]+R*y[6]+z*y[10],k},transformQuat(p,y,x){const k=x??new t(3),C=y[0],R=y[1],z=y[2],j=2*y[3],G=p[0],X=p[1],Z=p[2],ne=R*Z-z*X,oe=z*G-C*Z,le=C*X-R*G;return k[0]=G+ne*j+2*(R*le-z*oe),k[1]=X+oe*j+2*(z*ne-C*le),k[2]=Z+le*j+2*(C*oe-R*ne),k},getTranslation(p,y){const x=y??new t(3);return x[0]=p[12],x[1]=p[13],x[2]=p[14],x},getAxis(p,y,x){const k=x??new t(3),C=4*y;return k[0]=p[C+0],k[1]=p[C+1],k[2]=p[C+2],k},getScaling(p,y){const x=y??new t(3),k=p[0],C=p[1],R=p[2],z=p[4],j=p[5],G=p[6],X=p[8],Z=p[9],ne=p[10];return x[0]=Math.sqrt(k*k+C*C+R*R),x[1]=Math.sqrt(z*z+j*j+G*G),x[2]=Math.sqrt(X*X+Z*Z+ne*ne),x},rotateX(p,y,x,k){const C=k??new t(3),R=[],z=[];return R[0]=p[0]-y[0],R[1]=p[1]-y[1],R[2]=p[2]-y[2],z[0]=R[0],z[1]=R[1]*Math.cos(x)-R[2]*Math.sin(x),z[2]=R[1]*Math.sin(x)+R[2]*Math.cos(x),C[0]=z[0]+y[0],C[1]=z[1]+y[1],C[2]=z[2]+y[2],C},rotateY(p,y,x,k){const C=k??new t(3),R=[],z=[];return R[0]=p[0]-y[0],R[1]=p[1]-y[1],R[2]=p[2]-y[2],z[0]=R[2]*Math.sin(x)+R[0]*Math.cos(x),z[1]=R[1],z[2]=R[2]*Math.cos(x)-R[0]*Math.sin(x),C[0]=z[0]+y[0],C[1]=z[1]+y[1],C[2]=z[2]+y[2],C},rotateZ(p,y,x,k){const C=k??new t(3),R=[],z=[];return R[0]=p[0]-y[0],R[1]=p[1]-y[1],R[2]=p[2]-y[2],z[0]=R[0]*Math.cos(x)-R[1]*Math.sin(x),z[1]=R[0]*Math.sin(x)+R[1]*Math.cos(x),z[2]=R[2],C[0]=z[0]+y[0],C[1]=z[1]+y[1],C[2]=z[2]+y[2],C},setLength:g,truncate(p,y,x){const k=x??new t(3);return d(p)>y?g(p,y,k):v(p,k)},midpoint(p,y,x){return o(p,y,.5,x??new t(3))}}})(s),th.set(s,e)),e}const nh=new Map;function lT(s){let e=nh.get(s);return e||(e=(t=>{const n=nm(t),r=$o(t);function i(m,S,b){const f=b??new t(12);return f[0]=m[0]*S,f[1]=m[1]*S,f[2]=m[2]*S,f[4]=m[4]*S,f[5]=m[5]*S,f[6]=m[6]*S,f[8]=m[8]*S,f[9]=m[9]*S,f[10]=m[10]*S,f}const o=i;function a(m,S){const b=S??new t(12);return b[0]=m[0],b[1]=m[1],b[2]=m[2],b[4]=m[4],b[5]=m[5],b[6]=m[6],b[8]=m[8],b[9]=m[9],b[10]=m[10],b}const l=a;function u(m){const S=m??new t(12);return S[0]=1,S[1]=0,S[2]=0,S[4]=0,S[5]=1,S[6]=0,S[8]=0,S[9]=0,S[10]=1,S}function c(m,S){const b=S??new t(12),f=m[0],_=m[1],v=m[2],T=m[4],N=m[5],O=m[6],$=m[8],A=m[9],g=m[10],p=g*N-O*A,y=-g*T+O*$,x=A*T-N*$,k=1/(f*p+_*y+v*x);return b[0]=p*k,b[1]=(-g*_+v*A)*k,b[2]=(O*_-v*N)*k,b[4]=y*k,b[5]=(g*f-v*$)*k,b[6]=(-O*f+v*T)*k,b[8]=x*k,b[9]=(-A*f+_*$)*k,b[10]=(N*f-_*T)*k,b}const h=c;function d(m,S,b){const f=b??new t(12),_=m[0],v=m[1],T=m[2],N=m[4],O=m[5],$=m[6],A=m[8],g=m[9],p=m[10],y=S[0],x=S[1],k=S[2],C=S[4],R=S[5],z=S[6],j=S[8],G=S[9],X=S[10];return f[0]=_*y+N*x+A*k,f[1]=v*y+O*x+g*k,f[2]=T*y+$*x+p*k,f[4]=_*C+N*R+A*z,f[5]=v*C+O*R+g*z,f[6]=T*C+$*R+p*z,f[8]=_*j+N*G+A*X,f[9]=v*j+O*G+g*X,f[10]=T*j+$*G+p*X,f}const w=d;function I(m,S){const b=S??new t(12),f=Math.cos(m),_=Math.sin(m);return b[0]=f,b[1]=_,b[2]=0,b[4]=-_,b[5]=f,b[6]=0,b[8]=0,b[9]=0,b[10]=1,b}function E(m,S,b){const f=b??new t(12),_=m[0],v=m[1],T=m[2],N=m[4],O=m[5],$=m[6],A=Math.cos(S),g=Math.sin(S);return f[0]=A*_+g*N,f[1]=A*v+g*O,f[2]=A*T+g*$,f[4]=A*N-g*_,f[5]=A*O-g*v,f[6]=A*$-g*T,m!==f&&(f[8]=m[8],f[9]=m[9],f[10]=m[10]),f}return{add(m,S,b){const f=b??new t(12);return f[0]=m[0]+S[0],f[1]=m[1]+S[1],f[2]=m[2]+S[2],f[4]=m[4]+S[4],f[5]=m[5]+S[5],f[6]=m[6]+S[6],f[8]=m[8]+S[8],f[9]=m[9]+S[9],f[10]=m[10]+S[10],f},clone:l,copy:a,create(m,S,b,f,_,v,T,N,O){const $=new t(12);return $[3]=0,$[7]=0,$[11]=0,m!==void 0&&($[0]=m,S!==void 0&&($[1]=S,b!==void 0&&($[2]=b,f!==void 0&&($[4]=f,_!==void 0&&($[5]=_,v!==void 0&&($[6]=v,T!==void 0&&($[8]=T,N!==void 0&&($[9]=N,O!==void 0&&($[10]=O))))))))),$},determinant(m){const S=m[0],b=m[1],f=m[2],_=m[4],v=m[5],T=m[6],N=m[8],O=m[9],$=m[10];return S*(v*$-O*T)-_*(b*$-O*f)+N*(b*T-v*f)},equals(m,S){return m[0]===S[0]&&m[1]===S[1]&&m[2]===S[2]&&m[4]===S[4]&&m[5]===S[5]&&m[6]===S[6]&&m[8]===S[8]&&m[9]===S[9]&&m[10]===S[10]},equalsApproximately(m,S){return Math.abs(m[0]-S[0])<pe&&Math.abs(m[1]-S[1])<pe&&Math.abs(m[2]-S[2])<pe&&Math.abs(m[4]-S[4])<pe&&Math.abs(m[5]-S[5])<pe&&Math.abs(m[6]-S[6])<pe&&Math.abs(m[8]-S[8])<pe&&Math.abs(m[9]-S[9])<pe&&Math.abs(m[10]-S[10])<pe},fromMat4(m,S){const b=S??new t(12);return b[0]=m[0],b[1]=m[1],b[2]=m[2],b[3]=0,b[4]=m[4],b[5]=m[5],b[6]=m[6],b[7]=0,b[8]=m[8],b[9]=m[9],b[10]=m[10],b[11]=0,b},fromQuat(m,S){const b=S??new t(12),f=m[0],_=m[1],v=m[2],T=m[3],N=f+f,O=_+_,$=v+v,A=f*N,g=_*N,p=_*O,y=v*N,x=v*O,k=v*$,C=T*N,R=T*O,z=T*$;return b[0]=1-p-k,b[1]=g+z,b[2]=y-R,b[3]=0,b[4]=g-z,b[5]=1-A-k,b[6]=x+C,b[7]=0,b[8]=y+R,b[9]=x-C,b[10]=1-A-p,b[11]=0,b},get3DScaling(m,S){const b=S??r.create(),f=m[0],_=m[1],v=m[2],T=m[4],N=m[5],O=m[6],$=m[8],A=m[9],g=m[10];return b[0]=Math.sqrt(f*f+_*_+v*v),b[1]=Math.sqrt(T*T+N*N+O*O),b[2]=Math.sqrt($*$+A*A+g*g),b},getAxis(m,S,b){const f=b??n.create(),_=4*S;return f[0]=m[_+0],f[1]=m[_+1],f},getScaling(m,S){const b=S??n.create(),f=m[0],_=m[1],v=m[4],T=m[5];return b[0]=Math.sqrt(f*f+_*_),b[1]=Math.sqrt(v*v+T*T),b},getTranslation(m,S){const b=S??n.create();return b[0]=m[8],b[1]=m[9],b},identity:u,inverse:c,invert:h,mul:w,mulScalar:o,multiply:d,multiplyScalar:i,negate(m,S){const b=S??new t(12);return b[0]=-m[0],b[1]=-m[1],b[2]=-m[2],b[4]=-m[4],b[5]=-m[5],b[6]=-m[6],b[8]=-m[8],b[9]=-m[9],b[10]=-m[10],b},rotate:E,rotateX(m,S,b){const f=b??new t(12),_=m[4],v=m[5],T=m[6],N=m[8],O=m[9],$=m[10],A=Math.cos(S),g=Math.sin(S);return f[4]=A*_+g*N,f[5]=A*v+g*O,f[6]=A*T+g*$,f[8]=A*N-g*_,f[9]=A*O-g*v,f[10]=A*$-g*T,m!==f&&(f[0]=m[0],f[1]=m[1],f[2]=m[2]),f},rotateY(m,S,b){const f=b??new t(12),_=m[0],v=m[1],T=m[2],N=m[8],O=m[9],$=m[10],A=Math.cos(S),g=Math.sin(S);return f[0]=A*_-g*N,f[1]=A*v-g*O,f[2]=A*T-g*$,f[8]=A*N+g*_,f[9]=A*O+g*v,f[10]=A*$+g*T,m!==f&&(f[4]=m[4],f[5]=m[5],f[6]=m[6]),f},rotateZ:E,rotation:I,rotationX(m,S){const b=S??new t(12),f=Math.cos(m),_=Math.sin(m);return b[0]=1,b[1]=0,b[2]=0,b[4]=0,b[5]=f,b[6]=_,b[8]=0,b[9]=-_,b[10]=f,b},rotationY(m,S){const b=S??new t(12),f=Math.cos(m),_=Math.sin(m);return b[0]=f,b[1]=0,b[2]=-_,b[4]=0,b[5]=1,b[6]=0,b[8]=_,b[9]=0,b[10]=f,b},rotationZ:I,scale(m,S,b){const f=b??new t(12),_=S[0],v=S[1];return f[0]=_*m[0],f[1]=_*m[1],f[2]=_*m[2],f[4]=v*m[4],f[5]=v*m[5],f[6]=v*m[6],m!==f&&(f[8]=m[8],f[9]=m[9],f[10]=m[10]),f},scale3D(m,S,b){const f=b??new t(12),_=S[0],v=S[1],T=S[2];return f[0]=_*m[0],f[1]=_*m[1],f[2]=_*m[2],f[4]=v*m[4],f[5]=v*m[5],f[6]=v*m[6],f[8]=T*m[8],f[9]=T*m[9],f[10]=T*m[10],f},scaling(m,S){const b=S??new t(12);return b[0]=m[0],b[1]=0,b[2]=0,b[4]=0,b[5]=m[1],b[6]=0,b[8]=0,b[9]=0,b[10]=1,b},scaling3D(m,S){const b=S??new t(12);return b[0]=m[0],b[1]=0,b[2]=0,b[4]=0,b[5]=m[1],b[6]=0,b[8]=0,b[9]=0,b[10]=m[2],b},set(m,S,b,f,_,v,T,N,O,$){const A=$??new t(12);return A[0]=m,A[1]=S,A[2]=b,A[3]=0,A[4]=f,A[5]=_,A[6]=v,A[7]=0,A[8]=T,A[9]=N,A[10]=O,A[11]=0,A},setAxis(m,S,b,f){const _=f===m?m:a(m,f),v=4*b;return _[v+0]=S[0],_[v+1]=S[1],_},setTranslation(m,S,b){const f=b??u();return m!==f&&(f[0]=m[0],f[1]=m[1],f[2]=m[2],f[4]=m[4],f[5]=m[5],f[6]=m[6]),f[8]=S[0],f[9]=S[1],f[10]=1,f},translate(m,S,b){const f=b??new t(12),_=S[0],v=S[1],T=m[0],N=m[1],O=m[2],$=m[4],A=m[5],g=m[6],p=m[8],y=m[9],x=m[10];return m!==f&&(f[0]=T,f[1]=N,f[2]=O,f[4]=$,f[5]=A,f[6]=g),f[8]=T*_+$*v+p,f[9]=N*_+A*v+y,f[10]=O*_+g*v+x,f},translation(m,S){const b=S??new t(12);return b[0]=1,b[1]=0,b[2]=0,b[4]=0,b[5]=1,b[6]=0,b[8]=m[0],b[9]=m[1],b[10]=1,b},transpose(m,S){const b=S??new t(12);if(b===m){let p;return p=m[1],m[1]=m[4],m[4]=p,p=m[2],m[2]=m[8],m[8]=p,p=m[6],m[6]=m[9],m[9]=p,b}const f=m[0],_=m[1],v=m[2],T=m[4],N=m[5],O=m[6],$=m[8],A=m[9],g=m[10];return b[0]=f,b[1]=T,b[2]=$,b[4]=_,b[5]=N,b[6]=A,b[8]=v,b[9]=O,b[10]=g,b},uniformScale(m,S,b){const f=b??new t(12);return f[0]=S*m[0],f[1]=S*m[1],f[2]=S*m[2],f[4]=S*m[4],f[5]=S*m[5],f[6]=S*m[6],m!==f&&(f[8]=m[8],f[9]=m[9],f[10]=m[10]),f},uniformScale3D(m,S,b){const f=b??new t(12);return f[0]=S*m[0],f[1]=S*m[1],f[2]=S*m[2],f[4]=S*m[4],f[5]=S*m[5],f[6]=S*m[6],f[8]=S*m[8],f[9]=S*m[9],f[10]=S*m[10],f},uniformScaling(m,S){const b=S??new t(12);return b[0]=m,b[1]=0,b[2]=0,b[4]=0,b[5]=m,b[6]=0,b[8]=0,b[9]=0,b[10]=1,b},uniformScaling3D(m,S){const b=S??new t(12);return b[0]=m,b[1]=0,b[2]=0,b[4]=0,b[5]=m,b[6]=0,b[8]=0,b[9]=0,b[10]=m,b}}})(s),nh.set(s,e)),e}const sh=new Map;function uT(s){let e=sh.get(s);return e||(e=(t=>{const n=$o(t);function r(f,_,v){const T=v??new t(16);return T[0]=f[0]*_,T[1]=f[1]*_,T[2]=f[2]*_,T[3]=f[3]*_,T[4]=f[4]*_,T[5]=f[5]*_,T[6]=f[6]*_,T[7]=f[7]*_,T[8]=f[8]*_,T[9]=f[9]*_,T[10]=f[10]*_,T[11]=f[11]*_,T[12]=f[12]*_,T[13]=f[13]*_,T[14]=f[14]*_,T[15]=f[15]*_,T}const i=r;function o(f,_){const v=_??new t(16);return v[0]=f[0],v[1]=f[1],v[2]=f[2],v[3]=f[3],v[4]=f[4],v[5]=f[5],v[6]=f[6],v[7]=f[7],v[8]=f[8],v[9]=f[9],v[10]=f[10],v[11]=f[11],v[12]=f[12],v[13]=f[13],v[14]=f[14],v[15]=f[15],v}const a=o;function l(f){const _=f??new t(16);return _[0]=1,_[1]=0,_[2]=0,_[3]=0,_[4]=0,_[5]=1,_[6]=0,_[7]=0,_[8]=0,_[9]=0,_[10]=1,_[11]=0,_[12]=0,_[13]=0,_[14]=0,_[15]=1,_}function u(f,_){const v=_??new t(16),T=f[0],N=f[1],O=f[2],$=f[3],A=f[4],g=f[5],p=f[6],y=f[7],x=f[8],k=f[9],C=f[10],R=f[11],z=f[12],j=f[13],G=f[14],X=f[15],Z=C*X,ne=G*R,oe=p*X,le=G*y,Me=p*R,Fe=C*y,Ge=O*X,nt=G*$,st=O*R,rt=C*$,Tt=O*y,Et=p*$,At=x*j,Ct=z*k,$t=A*j,Nt=z*g,Dt=A*k,zr=x*g,Vr=T*j,Gr=z*N,Wr=T*k,qr=x*N,Hr=T*g,jr=A*N,Jl=Z*g+le*k+Me*j-(ne*g+oe*k+Fe*j),eu=ne*N+Ge*k+rt*j-(Z*N+nt*k+st*j),tu=oe*N+nt*g+Tt*j-(le*N+Ge*g+Et*j),nu=Fe*N+st*g+Et*k-(Me*N+rt*g+Tt*k),it=1/(T*Jl+A*eu+x*tu+z*nu);return v[0]=it*Jl,v[1]=it*eu,v[2]=it*tu,v[3]=it*nu,v[4]=it*(ne*A+oe*x+Fe*z-(Z*A+le*x+Me*z)),v[5]=it*(Z*T+nt*x+st*z-(ne*T+Ge*x+rt*z)),v[6]=it*(le*T+Ge*A+Et*z-(oe*T+nt*A+Tt*z)),v[7]=it*(Me*T+rt*A+Tt*x-(Fe*T+st*A+Et*x)),v[8]=it*(At*y+Nt*R+Dt*X-(Ct*y+$t*R+zr*X)),v[9]=it*(Ct*$+Vr*R+qr*X-(At*$+Gr*R+Wr*X)),v[10]=it*($t*$+Gr*y+Hr*X-(Nt*$+Vr*y+jr*X)),v[11]=it*(zr*$+Wr*y+jr*R-(Dt*$+qr*y+Hr*R)),v[12]=it*($t*C+zr*G+Ct*p-(Dt*G+At*p+Nt*C)),v[13]=it*(Wr*G+At*O+Gr*C-(Vr*C+qr*G+Ct*O)),v[14]=it*(Vr*p+jr*G+Nt*O-(Hr*G+$t*O+Gr*p)),v[15]=it*(Hr*C+Dt*O+qr*p-(Wr*p+jr*C+zr*O)),v}const c=u;function h(f,_,v){const T=v??new t(16),N=f[0],O=f[1],$=f[2],A=f[3],g=f[4],p=f[5],y=f[6],x=f[7],k=f[8],C=f[9],R=f[10],z=f[11],j=f[12],G=f[13],X=f[14],Z=f[15],ne=_[0],oe=_[1],le=_[2],Me=_[3],Fe=_[4],Ge=_[5],nt=_[6],st=_[7],rt=_[8],Tt=_[9],Et=_[10],At=_[11],Ct=_[12],$t=_[13],Nt=_[14],Dt=_[15];return T[0]=N*ne+g*oe+k*le+j*Me,T[1]=O*ne+p*oe+C*le+G*Me,T[2]=$*ne+y*oe+R*le+X*Me,T[3]=A*ne+x*oe+z*le+Z*Me,T[4]=N*Fe+g*Ge+k*nt+j*st,T[5]=O*Fe+p*Ge+C*nt+G*st,T[6]=$*Fe+y*Ge+R*nt+X*st,T[7]=A*Fe+x*Ge+z*nt+Z*st,T[8]=N*rt+g*Tt+k*Et+j*At,T[9]=O*rt+p*Tt+C*Et+G*At,T[10]=$*rt+y*Tt+R*Et+X*At,T[11]=A*rt+x*Tt+z*Et+Z*At,T[12]=N*Ct+g*$t+k*Nt+j*Dt,T[13]=O*Ct+p*$t+C*Nt+G*Dt,T[14]=$*Ct+y*$t+R*Nt+X*Dt,T[15]=A*Ct+x*$t+z*Nt+Z*Dt,T}const d=h,w=n.create(),I=n.create(),E=n.create();function m(f,_,v){const T=v??new t(16);let N=f[0],O=f[1],$=f[2];const A=Math.sqrt(N*N+O*O+$*$);N/=A,O/=A,$/=A;const g=N*N,p=O*O,y=$*$,x=Math.cos(_),k=Math.sin(_),C=1-x;return T[0]=g+(1-g)*x,T[1]=N*O*C+$*k,T[2]=N*$*C-O*k,T[3]=0,T[4]=N*O*C-$*k,T[5]=p+(1-p)*x,T[6]=O*$*C+N*k,T[7]=0,T[8]=N*$*C+O*k,T[9]=O*$*C-N*k,T[10]=y+(1-y)*x,T[11]=0,T[12]=0,T[13]=0,T[14]=0,T[15]=1,T}const S=m;function b(f,_,v,T){const N=T??new t(16);let O=_[0],$=_[1],A=_[2];const g=Math.sqrt(O*O+$*$+A*A);O/=g,$/=g,A/=g;const p=O*O,y=$*$,x=A*A,k=Math.cos(v),C=Math.sin(v),R=1-k,z=p+(1-p)*k,j=O*$*R+A*C,G=O*A*R-$*C,X=O*$*R-A*C,Z=y+(1-y)*k,ne=$*A*R+O*C,oe=O*A*R+$*C,le=$*A*R-O*C,Me=x+(1-x)*k,Fe=f[0],Ge=f[1],nt=f[2],st=f[3],rt=f[4],Tt=f[5],Et=f[6],At=f[7],Ct=f[8],$t=f[9],Nt=f[10],Dt=f[11];return N[0]=z*Fe+j*rt+G*Ct,N[1]=z*Ge+j*Tt+G*$t,N[2]=z*nt+j*Et+G*Nt,N[3]=z*st+j*At+G*Dt,N[4]=X*Fe+Z*rt+ne*Ct,N[5]=X*Ge+Z*Tt+ne*$t,N[6]=X*nt+Z*Et+ne*Nt,N[7]=X*st+Z*At+ne*Dt,N[8]=oe*Fe+le*rt+Me*Ct,N[9]=oe*Ge+le*Tt+Me*$t,N[10]=oe*nt+le*Et+Me*Nt,N[11]=oe*st+le*At+Me*Dt,f!==N&&(N[12]=f[12],N[13]=f[13],N[14]=f[14],N[15]=f[15]),N}return{add(f,_,v){const T=v??new t(16);return T[0]=f[0]+_[0],T[1]=f[1]+_[1],T[2]=f[2]+_[2],T[3]=f[3]+_[3],T[4]=f[4]+_[4],T[5]=f[5]+_[5],T[6]=f[6]+_[6],T[7]=f[7]+_[7],T[8]=f[8]+_[8],T[9]=f[9]+_[9],T[10]=f[10]+_[10],T[11]=f[11]+_[11],T[12]=f[12]+_[12],T[13]=f[13]+_[13],T[14]=f[14]+_[14],T[15]=f[15]+_[15],T},aim(f,_,v,T){const N=T??new t(16);return n.normalize(n.subtract(_,f,E),E),n.normalize(n.cross(v,E,w),w),n.normalize(n.cross(E,w,I),I),N[0]=w[0],N[1]=w[1],N[2]=w[2],N[3]=0,N[4]=I[0],N[5]=I[1],N[6]=I[2],N[7]=0,N[8]=E[0],N[9]=E[1],N[10]=E[2],N[11]=0,N[12]=f[0],N[13]=f[1],N[14]=f[2],N[15]=1,N},axisRotate:b,axisRotation:m,cameraAim(f,_,v,T){const N=T??new t(16);return n.normalize(n.subtract(f,_,E),E),n.normalize(n.cross(v,E,w),w),n.normalize(n.cross(E,w,I),I),N[0]=w[0],N[1]=w[1],N[2]=w[2],N[3]=0,N[4]=I[0],N[5]=I[1],N[6]=I[2],N[7]=0,N[8]=E[0],N[9]=E[1],N[10]=E[2],N[11]=0,N[12]=f[0],N[13]=f[1],N[14]=f[2],N[15]=1,N},clone:a,copy:o,create(f,_,v,T,N,O,$,A,g,p,y,x,k,C,R,z){const j=new t(16);return f!==void 0&&(j[0]=f,_!==void 0&&(j[1]=_,v!==void 0&&(j[2]=v,T!==void 0&&(j[3]=T,N!==void 0&&(j[4]=N,O!==void 0&&(j[5]=O,$!==void 0&&(j[6]=$,A!==void 0&&(j[7]=A,g!==void 0&&(j[8]=g,p!==void 0&&(j[9]=p,y!==void 0&&(j[10]=y,x!==void 0&&(j[11]=x,k!==void 0&&(j[12]=k,C!==void 0&&(j[13]=C,R!==void 0&&(j[14]=R,z!==void 0&&(j[15]=z)))))))))))))))),j},determinant(f){const _=f[0],v=f[1],T=f[2],N=f[3],O=f[4],$=f[5],A=f[6],g=f[7],p=f[8],y=f[9],x=f[10],k=f[11],C=f[12],R=f[13],z=f[14],j=f[15],G=x*j,X=z*k,Z=A*j,ne=z*g,oe=A*k,le=x*g,Me=T*j,Fe=z*N,Ge=T*k,nt=x*N,st=T*g,rt=A*N;return _*(G*$+ne*y+oe*R-(X*$+Z*y+le*R))+O*(X*v+Me*y+nt*R-(G*v+Fe*y+Ge*R))+p*(Z*v+Fe*$+st*R-(ne*v+Me*$+rt*R))+C*(le*v+Ge*$+rt*y-(oe*v+nt*$+st*y))},equals(f,_){return f[0]===_[0]&&f[1]===_[1]&&f[2]===_[2]&&f[3]===_[3]&&f[4]===_[4]&&f[5]===_[5]&&f[6]===_[6]&&f[7]===_[7]&&f[8]===_[8]&&f[9]===_[9]&&f[10]===_[10]&&f[11]===_[11]&&f[12]===_[12]&&f[13]===_[13]&&f[14]===_[14]&&f[15]===_[15]},equalsApproximately(f,_){return Math.abs(f[0]-_[0])<pe&&Math.abs(f[1]-_[1])<pe&&Math.abs(f[2]-_[2])<pe&&Math.abs(f[3]-_[3])<pe&&Math.abs(f[4]-_[4])<pe&&Math.abs(f[5]-_[5])<pe&&Math.abs(f[6]-_[6])<pe&&Math.abs(f[7]-_[7])<pe&&Math.abs(f[8]-_[8])<pe&&Math.abs(f[9]-_[9])<pe&&Math.abs(f[10]-_[10])<pe&&Math.abs(f[11]-_[11])<pe&&Math.abs(f[12]-_[12])<pe&&Math.abs(f[13]-_[13])<pe&&Math.abs(f[14]-_[14])<pe&&Math.abs(f[15]-_[15])<pe},fromMat3(f,_){const v=_??new t(16);return v[0]=f[0],v[1]=f[1],v[2]=f[2],v[3]=0,v[4]=f[4],v[5]=f[5],v[6]=f[6],v[7]=0,v[8]=f[8],v[9]=f[9],v[10]=f[10],v[11]=0,v[12]=0,v[13]=0,v[14]=0,v[15]=1,v},fromQuat(f,_){const v=_??new t(16),T=f[0],N=f[1],O=f[2],$=f[3],A=T+T,g=N+N,p=O+O,y=T*A,x=N*A,k=N*g,C=O*A,R=O*g,z=O*p,j=$*A,G=$*g,X=$*p;return v[0]=1-k-z,v[1]=x+X,v[2]=C-G,v[3]=0,v[4]=x-X,v[5]=1-y-z,v[6]=R+j,v[7]=0,v[8]=C+G,v[9]=R-j,v[10]=1-y-k,v[11]=0,v[12]=0,v[13]=0,v[14]=0,v[15]=1,v},frustum(f,_,v,T,N,O,$){const A=$??new t(16),g=_-f,p=T-v,y=N-O;return A[0]=2*N/g,A[1]=0,A[2]=0,A[3]=0,A[4]=0,A[5]=2*N/p,A[6]=0,A[7]=0,A[8]=(f+_)/g,A[9]=(T+v)/p,A[10]=O/y,A[11]=-1,A[12]=0,A[13]=0,A[14]=N*O/y,A[15]=0,A},frustumReverseZ(f,_,v,T,N,O=1/0,$){const A=$??new t(16),g=_-f,p=T-v;if(A[0]=2*N/g,A[1]=0,A[2]=0,A[3]=0,A[4]=0,A[5]=2*N/p,A[6]=0,A[7]=0,A[8]=(f+_)/g,A[9]=(T+v)/p,A[11]=-1,A[12]=0,A[13]=0,A[15]=0,O===1/0)A[10]=0,A[14]=N;else{const y=1/(O-N);A[10]=N*y,A[14]=O*N*y}return A},getAxis(f,_,v){const T=v??n.create(),N=4*_;return T[0]=f[N+0],T[1]=f[N+1],T[2]=f[N+2],T},getScaling(f,_){const v=_??n.create(),T=f[0],N=f[1],O=f[2],$=f[4],A=f[5],g=f[6],p=f[8],y=f[9],x=f[10];return v[0]=Math.sqrt(T*T+N*N+O*O),v[1]=Math.sqrt($*$+A*A+g*g),v[2]=Math.sqrt(p*p+y*y+x*x),v},getTranslation(f,_){const v=_??n.create();return v[0]=f[12],v[1]=f[13],v[2]=f[14],v},identity:l,inverse:u,invert:c,lookAt(f,_,v,T){const N=T??new t(16);return n.normalize(n.subtract(f,_,E),E),n.normalize(n.cross(v,E,w),w),n.normalize(n.cross(E,w,I),I),N[0]=w[0],N[1]=I[0],N[2]=E[0],N[3]=0,N[4]=w[1],N[5]=I[1],N[6]=E[1],N[7]=0,N[8]=w[2],N[9]=I[2],N[10]=E[2],N[11]=0,N[12]=-(w[0]*f[0]+w[1]*f[1]+w[2]*f[2]),N[13]=-(I[0]*f[0]+I[1]*f[1]+I[2]*f[2]),N[14]=-(E[0]*f[0]+E[1]*f[1]+E[2]*f[2]),N[15]=1,N},mul:d,mulScalar:i,multiply:h,multiplyScalar:r,negate(f,_){const v=_??new t(16);return v[0]=-f[0],v[1]=-f[1],v[2]=-f[2],v[3]=-f[3],v[4]=-f[4],v[5]=-f[5],v[6]=-f[6],v[7]=-f[7],v[8]=-f[8],v[9]=-f[9],v[10]=-f[10],v[11]=-f[11],v[12]=-f[12],v[13]=-f[13],v[14]=-f[14],v[15]=-f[15],v},ortho(f,_,v,T,N,O,$){const A=$??new t(16);return A[0]=2/(_-f),A[1]=0,A[2]=0,A[3]=0,A[4]=0,A[5]=2/(T-v),A[6]=0,A[7]=0,A[8]=0,A[9]=0,A[10]=1/(N-O),A[11]=0,A[12]=(_+f)/(f-_),A[13]=(T+v)/(v-T),A[14]=N/(N-O),A[15]=1,A},perspective(f,_,v,T,N){const O=N??new t(16),$=Math.tan(.5*Math.PI-.5*f);if(O[0]=$/_,O[1]=0,O[2]=0,O[3]=0,O[4]=0,O[5]=$,O[6]=0,O[7]=0,O[8]=0,O[9]=0,O[11]=-1,O[12]=0,O[13]=0,O[15]=0,Number.isFinite(T)){const A=1/(v-T);O[10]=T*A,O[14]=T*v*A}else O[10]=-1,O[14]=-v;return O},perspectiveReverseZ(f,_,v,T=1/0,N){const O=N??new t(16),$=1/Math.tan(.5*f);if(O[0]=$/_,O[1]=0,O[2]=0,O[3]=0,O[4]=0,O[5]=$,O[6]=0,O[7]=0,O[8]=0,O[9]=0,O[11]=-1,O[12]=0,O[13]=0,O[15]=0,T===1/0)O[10]=0,O[14]=v;else{const A=1/(T-v);O[10]=v*A,O[14]=T*v*A}return O},rotate:b,rotateX(f,_,v){const T=v??new t(16),N=f[4],O=f[5],$=f[6],A=f[7],g=f[8],p=f[9],y=f[10],x=f[11],k=Math.cos(_),C=Math.sin(_);return T[4]=k*N+C*g,T[5]=k*O+C*p,T[6]=k*$+C*y,T[7]=k*A+C*x,T[8]=k*g-C*N,T[9]=k*p-C*O,T[10]=k*y-C*$,T[11]=k*x-C*A,f!==T&&(T[0]=f[0],T[1]=f[1],T[2]=f[2],T[3]=f[3],T[12]=f[12],T[13]=f[13],T[14]=f[14],T[15]=f[15]),T},rotateY(f,_,v){const T=v??new t(16),N=f[0],O=f[1],$=f[2],A=f[3],g=f[8],p=f[9],y=f[10],x=f[11],k=Math.cos(_),C=Math.sin(_);return T[0]=k*N-C*g,T[1]=k*O-C*p,T[2]=k*$-C*y,T[3]=k*A-C*x,T[8]=k*g+C*N,T[9]=k*p+C*O,T[10]=k*y+C*$,T[11]=k*x+C*A,f!==T&&(T[4]=f[4],T[5]=f[5],T[6]=f[6],T[7]=f[7],T[12]=f[12],T[13]=f[13],T[14]=f[14],T[15]=f[15]),T},rotateZ(f,_,v){const T=v??new t(16),N=f[0],O=f[1],$=f[2],A=f[3],g=f[4],p=f[5],y=f[6],x=f[7],k=Math.cos(_),C=Math.sin(_);return T[0]=k*N+C*g,T[1]=k*O+C*p,T[2]=k*$+C*y,T[3]=k*A+C*x,T[4]=k*g-C*N,T[5]=k*p-C*O,T[6]=k*y-C*$,T[7]=k*x-C*A,f!==T&&(T[8]=f[8],T[9]=f[9],T[10]=f[10],T[11]=f[11],T[12]=f[12],T[13]=f[13],T[14]=f[14],T[15]=f[15]),T},rotation:S,rotationX(f,_){const v=_??new t(16),T=Math.cos(f),N=Math.sin(f);return v[0]=1,v[1]=0,v[2]=0,v[3]=0,v[4]=0,v[5]=T,v[6]=N,v[7]=0,v[8]=0,v[9]=-N,v[10]=T,v[11]=0,v[12]=0,v[13]=0,v[14]=0,v[15]=1,v},rotationY(f,_){const v=_??new t(16),T=Math.cos(f),N=Math.sin(f);return v[0]=T,v[1]=0,v[2]=-N,v[3]=0,v[4]=0,v[5]=1,v[6]=0,v[7]=0,v[8]=N,v[9]=0,v[10]=T,v[11]=0,v[12]=0,v[13]=0,v[14]=0,v[15]=1,v},rotationZ(f,_){const v=_??new t(16),T=Math.cos(f),N=Math.sin(f);return v[0]=T,v[1]=N,v[2]=0,v[3]=0,v[4]=-N,v[5]=T,v[6]=0,v[7]=0,v[8]=0,v[9]=0,v[10]=1,v[11]=0,v[12]=0,v[13]=0,v[14]=0,v[15]=1,v},scale(f,_,v){const T=v??new t(16),N=_[0],O=_[1],$=_[2];return T[0]=N*f[0],T[1]=N*f[1],T[2]=N*f[2],T[3]=N*f[3],T[4]=O*f[4],T[5]=O*f[5],T[6]=O*f[6],T[7]=O*f[7],T[8]=$*f[8],T[9]=$*f[9],T[10]=$*f[10],T[11]=$*f[11],f!==T&&(T[12]=f[12],T[13]=f[13],T[14]=f[14],T[15]=f[15]),T},scaling(f,_){const v=_??new t(16);return v[0]=f[0],v[1]=0,v[2]=0,v[3]=0,v[4]=0,v[5]=f[1],v[6]=0,v[7]=0,v[8]=0,v[9]=0,v[10]=f[2],v[11]=0,v[12]=0,v[13]=0,v[14]=0,v[15]=1,v},set(f,_,v,T,N,O,$,A,g,p,y,x,k,C,R,z,j){const G=j??new t(16);return G[0]=f,G[1]=_,G[2]=v,G[3]=T,G[4]=N,G[5]=O,G[6]=$,G[7]=A,G[8]=g,G[9]=p,G[10]=y,G[11]=x,G[12]=k,G[13]=C,G[14]=R,G[15]=z,G},setAxis(f,_,v,T){const N=T===f?T:o(f,T),O=4*v;return N[O+0]=_[0],N[O+1]=_[1],N[O+2]=_[2],N},setTranslation(f,_,v){const T=v??l();return f!==T&&(T[0]=f[0],T[1]=f[1],T[2]=f[2],T[3]=f[3],T[4]=f[4],T[5]=f[5],T[6]=f[6],T[7]=f[7],T[8]=f[8],T[9]=f[9],T[10]=f[10],T[11]=f[11]),T[12]=_[0],T[13]=_[1],T[14]=_[2],T[15]=1,T},translate(f,_,v){const T=v??new t(16),N=_[0],O=_[1],$=_[2],A=f[0],g=f[1],p=f[2],y=f[3],x=f[4],k=f[5],C=f[6],R=f[7],z=f[8],j=f[9],G=f[10],X=f[11],Z=f[12],ne=f[13],oe=f[14],le=f[15];return f!==T&&(T[0]=A,T[1]=g,T[2]=p,T[3]=y,T[4]=x,T[5]=k,T[6]=C,T[7]=R,T[8]=z,T[9]=j,T[10]=G,T[11]=X),T[12]=A*N+x*O+z*$+Z,T[13]=g*N+k*O+j*$+ne,T[14]=p*N+C*O+G*$+oe,T[15]=y*N+R*O+X*$+le,T},translation(f,_){const v=_??new t(16);return v[0]=1,v[1]=0,v[2]=0,v[3]=0,v[4]=0,v[5]=1,v[6]=0,v[7]=0,v[8]=0,v[9]=0,v[10]=1,v[11]=0,v[12]=f[0],v[13]=f[1],v[14]=f[2],v[15]=1,v},transpose(f,_){const v=_??new t(16);if(v===f){let Z;return Z=f[1],f[1]=f[4],f[4]=Z,Z=f[2],f[2]=f[8],f[8]=Z,Z=f[3],f[3]=f[12],f[12]=Z,Z=f[6],f[6]=f[9],f[9]=Z,Z=f[7],f[7]=f[13],f[13]=Z,Z=f[11],f[11]=f[14],f[14]=Z,v}const T=f[0],N=f[1],O=f[2],$=f[3],A=f[4],g=f[5],p=f[6],y=f[7],x=f[8],k=f[9],C=f[10],R=f[11],z=f[12],j=f[13],G=f[14],X=f[15];return v[0]=T,v[1]=A,v[2]=x,v[3]=z,v[4]=N,v[5]=g,v[6]=k,v[7]=j,v[8]=O,v[9]=p,v[10]=C,v[11]=G,v[12]=$,v[13]=y,v[14]=R,v[15]=X,v},uniformScale(f,_,v){const T=v??new t(16);return T[0]=_*f[0],T[1]=_*f[1],T[2]=_*f[2],T[3]=_*f[3],T[4]=_*f[4],T[5]=_*f[5],T[6]=_*f[6],T[7]=_*f[7],T[8]=_*f[8],T[9]=_*f[9],T[10]=_*f[10],T[11]=_*f[11],f!==T&&(T[12]=f[12],T[13]=f[13],T[14]=f[14],T[15]=f[15]),T},uniformScaling(f,_){const v=_??new t(16);return v[0]=f,v[1]=0,v[2]=0,v[3]=0,v[4]=0,v[5]=f,v[6]=0,v[7]=0,v[8]=0,v[9]=0,v[10]=f,v[11]=0,v[12]=0,v[13]=0,v[14]=0,v[15]=1,v}}})(s),sh.set(s,e)),e}const rh=new Map;function cT(s){let e=rh.get(s);return e||(e=(t=>{const n=$o(t);function r(g,p,y,x){const k=new t(4);return g!==void 0&&(k[0]=g,p!==void 0&&(k[1]=p,y!==void 0&&(k[2]=y,x!==void 0&&(k[3]=x)))),k}const i=r;function o(g,p,y){const x=y??new t(4),k=.5*p,C=Math.sin(k);return x[0]=C*g[0],x[1]=C*g[1],x[2]=C*g[2],x[3]=Math.cos(k),x}function a(g,p,y){const x=y??new t(4),k=g[0],C=g[1],R=g[2],z=g[3],j=p[0],G=p[1],X=p[2],Z=p[3];return x[0]=k*Z+z*j+C*X-R*G,x[1]=C*Z+z*G+R*j-k*X,x[2]=R*Z+z*X+k*G-C*j,x[3]=z*Z-k*j-C*G-R*X,x}const l=a;function u(g,p,y,x){const k=x??new t(4),C=g[0],R=g[1],z=g[2],j=g[3];let G,X,Z=p[0],ne=p[1],oe=p[2],le=p[3],Me=C*Z+R*ne+z*oe+j*le;if(Me<0&&(Me=-Me,Z=-Z,ne=-ne,oe=-oe,le=-le),1-Me>pe){const Fe=Math.acos(Me),Ge=Math.sin(Fe);G=Math.sin((1-y)*Fe)/Ge,X=Math.sin(y*Fe)/Ge}else G=1-y,X=y;return k[0]=G*C+X*Z,k[1]=G*R+X*ne,k[2]=G*z+X*oe,k[3]=G*j+X*le,k}function c(g,p){const y=p??new t(4);return y[0]=g[0],y[1]=g[1],y[2]=g[2],y[3]=g[3],y}const h=c;function d(g,p,y){const x=y??new t(4);return x[0]=g[0]-p[0],x[1]=g[1]-p[1],x[2]=g[2]-p[2],x[3]=g[3]-p[3],x}const w=d;function I(g,p,y){const x=y??new t(4);return x[0]=g[0]*p,x[1]=g[1]*p,x[2]=g[2]*p,x[3]=g[3]*p,x}const E=I;function m(g,p){return g[0]*p[0]+g[1]*p[1]+g[2]*p[2]+g[3]*p[3]}function S(g){const p=g[0],y=g[1],x=g[2],k=g[3];return Math.sqrt(p*p+y*y+x*x+k*k)}const b=S;function f(g){const p=g[0],y=g[1],x=g[2],k=g[3];return p*p+y*y+x*x+k*k}const _=f;function v(g,p){const y=p??new t(4),x=g[0],k=g[1],C=g[2],R=g[3],z=Math.sqrt(x*x+k*k+C*C+R*R);return z>1e-5?(y[0]=x/z,y[1]=k/z,y[2]=C/z,y[3]=R/z):(y[0]=0,y[1]=0,y[2]=0,y[3]=1),y}const T=n.create(),N=n.create(),O=n.create(),$=new t(4),A=new t(4);return{create:r,fromValues:i,set(g,p,y,x,k){const C=k??new t(4);return C[0]=g,C[1]=p,C[2]=y,C[3]=x,C},fromAxisAngle:o,toAxisAngle(g,p){const y=p??n.create(3),x=2*Math.acos(g[3]),k=Math.sin(.5*x);return k>pe?(y[0]=g[0]/k,y[1]=g[1]/k,y[2]=g[2]/k):(y[0]=1,y[1]=0,y[2]=0),{angle:x,axis:y}},angle(g,p){const y=m(g,p);return Math.acos(2*y*y-1)},multiply:a,mul:l,rotateX(g,p,y){const x=y??new t(4),k=.5*p,C=g[0],R=g[1],z=g[2],j=g[3],G=Math.sin(k),X=Math.cos(k);return x[0]=C*X+j*G,x[1]=R*X+z*G,x[2]=z*X-R*G,x[3]=j*X-C*G,x},rotateY(g,p,y){const x=y??new t(4),k=.5*p,C=g[0],R=g[1],z=g[2],j=g[3],G=Math.sin(k),X=Math.cos(k);return x[0]=C*X-z*G,x[1]=R*X+j*G,x[2]=z*X+C*G,x[3]=j*X-R*G,x},rotateZ(g,p,y){const x=y??new t(4),k=.5*p,C=g[0],R=g[1],z=g[2],j=g[3],G=Math.sin(k),X=Math.cos(k);return x[0]=C*X+R*G,x[1]=R*X-C*G,x[2]=z*X+j*G,x[3]=j*X-z*G,x},slerp:u,inverse(g,p){const y=p??new t(4),x=g[0],k=g[1],C=g[2],R=g[3],z=x*x+k*k+C*C+R*R,j=z?1/z:0;return y[0]=-x*j,y[1]=-k*j,y[2]=-C*j,y[3]=R*j,y},conjugate(g,p){const y=p??new t(4);return y[0]=-g[0],y[1]=-g[1],y[2]=-g[2],y[3]=g[3],y},fromMat(g,p){const y=p??new t(4),x=g[0]+g[5]+g[10];if(x>0){const k=Math.sqrt(x+1);y[3]=.5*k;const C=.5/k;y[0]=(g[6]-g[9])*C,y[1]=(g[8]-g[2])*C,y[2]=(g[1]-g[4])*C}else{let k=0;g[5]>g[0]&&(k=1),g[10]>g[4*k+k]&&(k=2);const C=(k+1)%3,R=(k+2)%3,z=Math.sqrt(g[4*k+k]-g[4*C+C]-g[4*R+R]+1);y[k]=.5*z;const j=.5/z;y[3]=(g[4*C+R]-g[4*R+C])*j,y[C]=(g[4*C+k]+g[4*k+C])*j,y[R]=(g[4*R+k]+g[4*k+R])*j}return y},fromEuler(g,p,y,x,k){const C=k??new t(4),R=.5*g,z=.5*p,j=.5*y,G=Math.sin(R),X=Math.cos(R),Z=Math.sin(z),ne=Math.cos(z),oe=Math.sin(j),le=Math.cos(j);switch(x){case"xyz":C[0]=G*ne*le+X*Z*oe,C[1]=X*Z*le-G*ne*oe,C[2]=X*ne*oe+G*Z*le,C[3]=X*ne*le-G*Z*oe;break;case"xzy":C[0]=G*ne*le-X*Z*oe,C[1]=X*Z*le-G*ne*oe,C[2]=X*ne*oe+G*Z*le,C[3]=X*ne*le+G*Z*oe;break;case"yxz":C[0]=G*ne*le+X*Z*oe,C[1]=X*Z*le-G*ne*oe,C[2]=X*ne*oe-G*Z*le,C[3]=X*ne*le+G*Z*oe;break;case"yzx":C[0]=G*ne*le+X*Z*oe,C[1]=X*Z*le+G*ne*oe,C[2]=X*ne*oe-G*Z*le,C[3]=X*ne*le-G*Z*oe;break;case"zxy":C[0]=G*ne*le-X*Z*oe,C[1]=X*Z*le+G*ne*oe,C[2]=X*ne*oe+G*Z*le,C[3]=X*ne*le-G*Z*oe;break;case"zyx":C[0]=G*ne*le-X*Z*oe,C[1]=X*Z*le+G*ne*oe,C[2]=X*ne*oe-G*Z*le,C[3]=X*ne*le+G*Z*oe;break;default:throw new Error(`Unknown rotation order: ${x}`)}return C},copy:c,clone:h,add(g,p,y){const x=y??new t(4);return x[0]=g[0]+p[0],x[1]=g[1]+p[1],x[2]=g[2]+p[2],x[3]=g[3]+p[3],x},subtract:d,sub:w,mulScalar:I,scale:E,divScalar(g,p,y){const x=y??new t(4);return x[0]=g[0]/p,x[1]=g[1]/p,x[2]=g[2]/p,x[3]=g[3]/p,x},dot:m,lerp(g,p,y,x){const k=x??new t(4);return k[0]=g[0]+y*(p[0]-g[0]),k[1]=g[1]+y*(p[1]-g[1]),k[2]=g[2]+y*(p[2]-g[2]),k[3]=g[3]+y*(p[3]-g[3]),k},length:S,len:b,lengthSq:f,lenSq:_,normalize:v,equalsApproximately(g,p){return Math.abs(g[0]-p[0])<pe&&Math.abs(g[1]-p[1])<pe&&Math.abs(g[2]-p[2])<pe&&Math.abs(g[3]-p[3])<pe},equals(g,p){return g[0]===p[0]&&g[1]===p[1]&&g[2]===p[2]&&g[3]===p[3]},identity(g){const p=g??new t(4);return p[0]=0,p[1]=0,p[2]=0,p[3]=1,p},rotationTo(g,p,y){const x=y??new t(4),k=n.dot(g,p);return k<-.999999?(n.cross(N,g,T),n.len(T)<1e-6&&n.cross(O,g,T),n.normalize(T,T),o(T,Math.PI,x),x):k>.999999?(x[0]=0,x[1]=0,x[2]=0,x[3]=1,x):(n.cross(g,p,T),x[0]=T[0],x[1]=T[1],x[2]=T[2],x[3]=1+k,v(x,x))},sqlerp(g,p,y,x,k,C){const R=C??new t(4);return u(g,x,k,$),u(p,y,k,A),u($,A,2*k*(1-k),R),R}}})(s),rh.set(s,e)),e}const ih=new Map;function hT(s){let e=ih.get(s);return e||(e=(t=>{function n(g,p,y,x){const k=new t(4);return g!==void 0&&(k[0]=g,p!==void 0&&(k[1]=p,y!==void 0&&(k[2]=y,x!==void 0&&(k[3]=x)))),k}function r(g,p,y){const x=y??new t(4);return x[0]=g[0]-p[0],x[1]=g[1]-p[1],x[2]=g[2]-p[2],x[3]=g[3]-p[3],x}const i=r;function o(g,p,y,x){const k=x??new t(4);return k[0]=g[0]+y*(p[0]-g[0]),k[1]=g[1]+y*(p[1]-g[1]),k[2]=g[2]+y*(p[2]-g[2]),k[3]=g[3]+y*(p[3]-g[3]),k}function a(g,p,y){const x=y??new t(4);return x[0]=g[0]*p,x[1]=g[1]*p,x[2]=g[2]*p,x[3]=g[3]*p,x}const l=a;function u(g,p){const y=p??new t(4);return y[0]=1/g[0],y[1]=1/g[1],y[2]=1/g[2],y[3]=1/g[3],y}const c=u;function h(g){const p=g[0],y=g[1],x=g[2],k=g[3];return Math.sqrt(p*p+y*y+x*x+k*k)}const d=h;function w(g){const p=g[0],y=g[1],x=g[2],k=g[3];return p*p+y*y+x*x+k*k}const I=w;function E(g,p){const y=g[0]-p[0],x=g[1]-p[1],k=g[2]-p[2],C=g[3]-p[3];return Math.sqrt(y*y+x*x+k*k+C*C)}const m=E;function S(g,p){const y=g[0]-p[0],x=g[1]-p[1],k=g[2]-p[2],C=g[3]-p[3];return y*y+x*x+k*k+C*C}const b=S;function f(g,p){const y=p??new t(4),x=g[0],k=g[1],C=g[2],R=g[3],z=Math.sqrt(x*x+k*k+C*C+R*R);return z>1e-5?(y[0]=x/z,y[1]=k/z,y[2]=C/z,y[3]=R/z):(y[0]=0,y[1]=0,y[2]=0,y[3]=0),y}function _(g,p){const y=p??new t(4);return y[0]=g[0],y[1]=g[1],y[2]=g[2],y[3]=g[3],y}const v=_;function T(g,p,y){const x=y??new t(4);return x[0]=g[0]*p[0],x[1]=g[1]*p[1],x[2]=g[2]*p[2],x[3]=g[3]*p[3],x}const N=T;function O(g,p,y){const x=y??new t(4);return x[0]=g[0]/p[0],x[1]=g[1]/p[1],x[2]=g[2]/p[2],x[3]=g[3]/p[3],x}const $=O;function A(g,p,y){const x=y??new t(4);return f(g,x),a(x,p,x)}return{create:n,fromValues:n,set(g,p,y,x,k){const C=k??new t(4);return C[0]=g,C[1]=p,C[2]=y,C[3]=x,C},ceil(g,p){const y=p??new t(4);return y[0]=Math.ceil(g[0]),y[1]=Math.ceil(g[1]),y[2]=Math.ceil(g[2]),y[3]=Math.ceil(g[3]),y},floor(g,p){const y=p??new t(4);return y[0]=Math.floor(g[0]),y[1]=Math.floor(g[1]),y[2]=Math.floor(g[2]),y[3]=Math.floor(g[3]),y},round(g,p){const y=p??new t(4);return y[0]=Math.round(g[0]),y[1]=Math.round(g[1]),y[2]=Math.round(g[2]),y[3]=Math.round(g[3]),y},clamp(g,p=0,y=1,x){const k=x??new t(4);return k[0]=Math.min(y,Math.max(p,g[0])),k[1]=Math.min(y,Math.max(p,g[1])),k[2]=Math.min(y,Math.max(p,g[2])),k[3]=Math.min(y,Math.max(p,g[3])),k},add(g,p,y){const x=y??new t(4);return x[0]=g[0]+p[0],x[1]=g[1]+p[1],x[2]=g[2]+p[2],x[3]=g[3]+p[3],x},addScaled(g,p,y,x){const k=x??new t(4);return k[0]=g[0]+p[0]*y,k[1]=g[1]+p[1]*y,k[2]=g[2]+p[2]*y,k[3]=g[3]+p[3]*y,k},subtract:r,sub:i,equalsApproximately(g,p){return Math.abs(g[0]-p[0])<pe&&Math.abs(g[1]-p[1])<pe&&Math.abs(g[2]-p[2])<pe&&Math.abs(g[3]-p[3])<pe},equals(g,p){return g[0]===p[0]&&g[1]===p[1]&&g[2]===p[2]&&g[3]===p[3]},lerp:o,lerpV(g,p,y,x){const k=x??new t(4);return k[0]=g[0]+y[0]*(p[0]-g[0]),k[1]=g[1]+y[1]*(p[1]-g[1]),k[2]=g[2]+y[2]*(p[2]-g[2]),k[3]=g[3]+y[3]*(p[3]-g[3]),k},max(g,p,y){const x=y??new t(4);return x[0]=Math.max(g[0],p[0]),x[1]=Math.max(g[1],p[1]),x[2]=Math.max(g[2],p[2]),x[3]=Math.max(g[3],p[3]),x},min(g,p,y){const x=y??new t(4);return x[0]=Math.min(g[0],p[0]),x[1]=Math.min(g[1],p[1]),x[2]=Math.min(g[2],p[2]),x[3]=Math.min(g[3],p[3]),x},mulScalar:a,scale:l,divScalar(g,p,y){const x=y??new t(4);return x[0]=g[0]/p,x[1]=g[1]/p,x[2]=g[2]/p,x[3]=g[3]/p,x},inverse:u,invert:c,dot(g,p){return g[0]*p[0]+g[1]*p[1]+g[2]*p[2]+g[3]*p[3]},length:h,len:d,lengthSq:w,lenSq:I,distance:E,dist:m,distanceSq:S,distSq:b,normalize:f,negate(g,p){const y=p??new t(4);return y[0]=-g[0],y[1]=-g[1],y[2]=-g[2],y[3]=-g[3],y},copy:_,clone:v,multiply:T,mul:N,divide:O,div:$,zero(g){const p=g??new t(4);return p[0]=0,p[1]=0,p[2]=0,p[3]=0,p},transformMat4(g,p,y){const x=y??new t(4),k=g[0],C=g[1],R=g[2],z=g[3];return x[0]=p[0]*k+p[4]*C+p[8]*R+p[12]*z,x[1]=p[1]*k+p[5]*C+p[9]*R+p[13]*z,x[2]=p[2]*k+p[6]*C+p[10]*R+p[14]*z,x[3]=p[3]*k+p[7]*C+p[11]*R+p[15]*z,x},setLength:A,truncate(g,p,y){const x=y??new t(4);return h(g)>p?A(g,p,x):_(g,x)},midpoint(g,p,y){return o(g,p,.5,y??new t(4))}}})(s),ih.set(s,e)),e}function Fa(s,e,t,n,r,i){return{mat3:lT(s),mat4:uT(e),quat:cT(t),vec2:nm(n),vec3:$o(r),vec4:hT(i)}}const{mat3:oh,mat4:ah}=Fa(Float32Array,Float32Array,Float32Array,Float32Array,Float32Array,Float32Array);Fa(Float64Array,Float64Array,Float64Array,Float64Array,Float64Array,Float64Array),Fa(aT,Array,Array,Array,Array,Array);let fT=class extends Jp{#n;#t=!1;#e=oh.identity();#s=new Float32Array(3);#i=!1;#r;#a;#u=ah.identity();#o;#f;#h;#l;#c;#d;#g;#p=[0,0,0,0];#m=[];#b;#x;#v;#_=[0,0,0,0];#w;#y;constructor(s,e,t,n){super(s,e,"Render"),!t&&ie(te.CANVAS_NOT_FOUND);const r=t.getContext("webgpu");!r&&ie(te.CONTEXT_NOT_FOUND),r.configure({device:s,...n}),this.#o=this.CreateBuffer({size:this.#s.length*Float32Array.BYTES_PER_ELEMENT,label:"Render Pipeline Resolution Buffer",usage:gt.UNIFORM}),this.#r=t,this.#a=r,this.#S(),this.#g=n.format,this.CreatePassDescriptor(this.CreateColorAttachment())}ConfigureContext(s){const e=s.format??this.#g;this.#a.configure({device:this.Device,format:e,...s})}#S(){this.#s.set([this.#r.width,this.#r.height,this.DevicePixelRatio]),this.WriteBuffer(this.#o,this.#s)}#I(s){return s instanceof Co?s.rgba:s}CreateColorAttachment(s,e="clear",t="store",n,r,i){return{view:s,loadOp:e,storeOp:t,clearValue:n&&this.#I(n),resolveTarget:r,depthSlice:i}}CreateDepthAttachment(s,e=1,t="clear",n="store",r){return this.#i=!0,this.#f=new Zl(this.Device),{view:s,depthClearValue:e,depthLoadOp:t,depthStoreOp:n,depthReadOnly:r}}CreateStencilAttachment(s,e="clear",t="store",n){return{stencilClearValue:s,stencilLoadOp:e,stencilStoreOp:t,stencilReadOnly:n}}CreatePassDescriptor(s,e,t,n,r,i){const o=Array.isArray(s)&&s||[s];return this.#t=!o.some(({view:a})=>!!a),e??=this.CreatePipelineLabel("Render Pass"),this.Descriptor={colorAttachments:o,depthStencilAttachment:t,occlusionQuerySet:n,timestampWrites:r,maxDrawCount:i,label:e}}CreateIndexBuffer(s,e){const t=e?.label??"Index Buffer";return s=Array.isArray(s)&&new Uint32Array(s)||s,this.CreateBuffer({label:t,size:s.byteLength,usage:gt.INDEX,...e})}CreateVertexBufferAttribute(s,e=0,t=0){return this.#k(s,e,t)}#k(s,e=0,t=0){return{format:s,shaderLocation:e,offset:t}}CreateVertexBufferLayout(s,e,t="vertex"){!this.Reflect&&ie(te.SHADER_MODULE_NOT_FOUND,"`LegacyRenderer.CreateVertexBufferLayout`.\n            Call `LegacyRenderer.CreateShaderModule` before creating a vertex layout or vertex buffer.");const{entry:{vertex:n}}=this.Reflect,r=n.find(({name:a})=>t===a);!r&&ie(te.VERTEX_ENTRY_NOT_FOUND,`\`${t}\` in vertex shader entries.`);let i=[],o=0;for(let a=0,l=(s=Array.isArray(s)&&s||[s]).length;a<l;++a){const u=s[a],c=typeof u=="string",h=c?u:u.name,d=r.inputs.find(({name:w})=>h===w);if(d){const w=c?im(d.type.size):u.format;i.push(this.#k(w,+d.location,o)),o+=om(w);continue}mt(te.VERTEX_ATTRIBUTE_NOT_FOUND,`\`${h}\` in vertex shader inputs.`)}return{arrayStride:o,stepMode:e,attributes:i}}CreateVertexBuffer(s,e=1,t,n="vertex"){const r=e.label??"Vertex Buffer";if(s instanceof Float32Array)return this.CreateBuffer({label:r,size:s.byteLength,usage:gt.VERTEX,...e});const i=this.CreateVertexBufferLayout(s,t,n),o=(typeof e=="number"&&e||(e.count??1))*i.arrayStride;return{buffer:this.CreateBuffer({label:r,size:o,usage:gt.VERTEX,...e}),layout:i}}CreateVertexState(s,e="vertex",t,n){return{module:s,entryPoint:e,buffers:t=Array.isArray(t)&&t||[t],constants:n}}CreateBlendComponent(s="add",e="one",t="zero"){return{operation:s,srcFactor:e,dstFactor:t}}CreateTargetState(s=this.#g,e,t){return e&&={color:e.color??{},alpha:e.alpha??{}},{format:s,blend:e,writeMask:t}}CreateFragmentState(s,e="fragment",t,n){return t??=this.CreateTargetState(),{module:s,entryPoint:e,targets:t=Array.isArray(t)&&t||[t],constants:n}}CreateStencilFaceState(s,e,t,n){return{compare:s,failOp:e,depthFailOp:t,passOp:n}}CreateDepthStencilState(s="depth24plus",e=!0,t="less",n,r,i,o,a,l,u){return{format:s,depthWriteEnabled:e,depthCompare:t,stencilFront:n,stencilBack:r,stencilReadMask:i,stencilWriteMask:o,depthBias:a,depthBiasSlopeScale:l,depthBiasClamp:u}}CreateMultisampleState(s=4,e,t){return{count:s,mask:e,alphaToCoverageEnabled:t}}CreateStorageTextureBindingLayout(s=this.#g,e,t,n,r){return{binding:r,visibility:GPUShaderStage.FRAGMENT,storageTexture:{access:e,format:s,viewDimension:t}}}CreatePipeline(s={},e){let t=this.GetShaderModule(s),{vertex:n,fragment:r}=s;!t&&!n&&(t=this.CreateShaderModule()),t&&(n??=this.CreateVertexState(t),r??=this.CreateFragmentState(t));const i=s.label??this.CreatePipelineLabel("Render Pipeline"),o=s.layout??"auto";return this.SetPipeline(this.Device.createRenderPipeline({label:i,layout:o,vertex:n,fragment:r,...s})),e&&(this.#c?this.#c.setPipeline(this.Pipeline):mt(te.RENDER_PASS_NOT_FOUND)),this.Pipeline}SavePipelineState(){super.SavePipelineState(),this.#y=this.#d,this.#b=this.#m,this.#_=this.#p,this.#w=this.#l,this.#x=this.#t,this.#v=this.#i,this.#y&&=Object.values(this.#y)}ResetPipelineState(){super.ResetPipelineState(),this.SetVertexBuffers([]),this.SetIndexBuffer(void 0),this.#t=!1,this.#_=[0,0,0,0],this.#i=!1,this.#l=this.#l?.destroy()}RestorePipelineState(){super.RestorePipelineState(),this.#m=this.#b,this.#p=this.#_,this.#l=this.#w,this.#t=this.#x,this.#i=this.#v,this.SetIndexBuffer(...Array.isArray(this.#y)&&this.#y||[void 0])}SetVertexBuffers(s,e,t){e=Array.isArray(e)&&e||[e],t=Array.isArray(t)&&t||[t],this.#m=Array.isArray(s)&&s.map((n,r)=>({buffer:n,offset:e[r],size:t[r]}))||[{buffer:s,offset:e[0],size:t[0]}]}AddVertexBuffers(s,e,t){e=Array.isArray(e)&&e||[e],t=Array.isArray(t)&&t||[t],this.#m.push(...Array.isArray(s)&&s.map((n,r)=>({buffer:n,offset:e[r],size:t[r]}))||[{buffer:s,offset:e[0],size:t[0]}])}SetIndexBuffer(s,e="uint32",t,n){this.#d=s&&{buffer:s,format:e,offset:t,size:n}}SetCanvasSize(s,e,t=!0){!this.Device&&ie(te.DEVICE_NOT_FOUND),!this.#r&&ie(te.CANVAS_NOT_FOUND);let n=this.DevicePixelRatio*s|0,r=this.DevicePixelRatio*e|0;const i=this.Device.limits.maxTextureDimension2D;n=Math.max(1,Math.min(n,i)),r=Math.max(1,Math.min(r,i)),this.#r.width===n&&this.#r.height===r||(this.#r.height=r,this.#r.width=n,this.#S(),t&&(this.#r.style.width=`${s}px`,this.#r.style.height=`${e}px`))}SetTextureView(s,e=0){this.Descriptor.colorAttachments[e].view=s,this.#t=!s}UpdateOrthographicProjection(s=1,e=1e3,t=0,n=this.#r.clientWidth,r=this.#r.clientHeight,i=0){return ah.ortho(i,n,r,t,s,e,this.#u),this.#u}UpdateProjection2D(s=this.#r.clientWidth,e=this.#r.clientHeight){return oh.set(2/s,0,0,0,-2/e,0,-1,1,1,this.#e),this.#e}#T(){const s=this.CurrentTexture,{width:e,height:t}=s;this.#h&&this.#h.width===e&&this.#h.height===t||(this.#h?.destroy(),this.#h=this.#f.CreateTextureFromSource(s,{sampleCount:this.#l?.sampleCount??1,usage:GPUTextureUsage.RENDER_ATTACHMENT,label:"Depth Texture",format:"depth24plus",mipmaps:!1})),this.Descriptor.depthStencilAttachment.view=this.#h.createView()}Render(s,e=!0){this.#i&&this.#T(),this.#c||(this.#l?(this.Descriptor.colorAttachments[0].view=this.#l.createView(),this.Descriptor.colorAttachments[0].resolveTarget=this.CurrentTextureView):this.#t&&(this.Descriptor.colorAttachments[0].view=this.CurrentTextureView),this.#c=this.GetCommandEncoder().beginRenderPass(this.Descriptor),this.#c.setPipeline(this.Pipeline),this.#n=this.#d?this.#c.drawIndexed.bind(this.#c):this.#c.draw.bind(this.#c));for(let t=0,n=this.#m.length;t<n;++t){const{buffer:r,offset:i,size:o}=this.#m[t];this.#c.setVertexBuffer(t,r,i,o)}this.#d&&this.#c.setIndexBuffer(this.#d.buffer,this.#d.format,this.#d.offset,this.#d.size);for(let t=0,n=0,r=this.BindGroups.length;t<r;++t){const{bindGroup:i,dynamicOffsets:o,active:a}=this.BindGroups[t];a&&this.#c.setBindGroup(n++,i,o)}this.#c.setBlendConstant(this.#p),this.#n(...Array.isArray(s)&&s||[s]),e&&this.Submit()}DestroyCurrentPass(){this.#c?.end(),this.#c=void 0}Submit(){this.DestroyCurrentPass(),this.SubmitCommandBuffer(),this.SetCommandEncoder(void 0)}get Canvas(){return this.#r}get Context(){return this.#a}get CurrentPass(){return this.#c}get AspectRatio(){return!this.#r&&ie(te.CANVAS_NOT_FOUND),this.#r.width/this.#r.height}get Projection2D(){return this.#e}get DepthTexture(){return this.#h}get CurrentTexture(){return this.#a.getCurrentTexture()}get CurrentTextureView(){return this.CurrentTexture.createView()}get OrthographicProjection(){return this.#u}set MultisampleTexture(s){this.#l=s}get MultisampleTexture(){return this.#l}set BlendConstant(s){this.#p=this.#I(s)}get BlendConstant(){return this.#p}get ResolutionBuffer(){return this.#o}get DevicePixelRatio(){return globalThis.devicePixelRatio??1}set TextureView(s){this.Descriptor.colorAttachments[0].view=s,this.#t=!s}get BaseCanvasSize(){const{width:s,height:e}=this.#r,t=this.DevicePixelRatio;return[s/t,e/t]}get CanvasSize(){return[this.#r.width,this.#r.height]}Destroy(){super.Destroy(),this.DestroyCurrentPass(),this.#o.destroy(),this.#p=[0,0,0,0],this.#f=this.#f?.Destroy(),this.#h=this.#h?.destroy(),this.#w=this.#w?.destroy(),this.#x=this.#v=!1,this.#b?.forEach(({buffer:s})=>s.destroy()),this.#m.forEach(({buffer:s})=>s.destroy()),this.#d?.buffer.destroy(),this.#y?.splice(0),this.#y=void 0,this.#b?.splice(0),this.#b=void 0,this.#m.splice(0),this.ResetPipelineState(),this.#a.unconfigure()}},dT=class extends Jp{#n=[1];constructor(s,e){super(s,e,"Compute"),this.CreatePassDescriptor()}CreatePassDescriptor(s,e){return s??=this.CreatePipelineLabel("Compute Pass"),this.Descriptor={label:s,timestampWrites:e}}CreatePipeline(s){const e=s.label??this.CreatePipelineLabel("Compute Pipeline"),t=s.layout??"auto",n=this.GetShaderModule(s)??this.CreateShaderModule();return this.SetPipeline(this.Device.createComputePipeline({label:e,layout:t,compute:{module:n,...s}}))}Compute(s=!1){const e=this.GetCommandEncoder().beginComputePass(this.Descriptor);e.setPipeline(this.Pipeline);for(let t=0,n=0,r=this.BindGroups.length;t<r;++t){const{bindGroup:i,dynamicOffsets:o,active:a}=this.BindGroups[t];a&&e.setBindGroup(n++,i,o)}e.dispatchWorkgroups(...this.#n),e.end(),s&&this.Submit()}Submit(){this.SubmitCommandBuffer(),this.SetCommandEncoder(void 0)}set Workgroups(s){this.#n=Array.isArray(s)&&s||[s]}Destroy(){super.Destroy(),this.Workgroups=1}};class sm{Pipelines=[];#n;Device;#t;#e;#s;constructor(e,t,n){this.#s=n,this.#e=t,this.Device=e,this.#t=this.CreateStageLabel("Command Encoder")}CreateStageLabel(e){return this.#s&&e&&`${this.#s} ${e}`||""}#i(){return this.#e==="Render"&&GPUShaderStage.FRAGMENT||GPUShaderStage.COMPUTE}CreateTimestampWrites(e,t=0,n=1){return{querySet:e,beginningOfPassWriteIndex:t,endOfPassWriteIndex:n}}ResolveQuerySet(e,t,n=0,r=e.count,i=0){this.GetCommandEncoder(!0).resolveQuerySet(e,n,r,t,i)}CreateBufferBindingLayout(e,t,n,r,i){return r??=this.#i(),{binding:i,visibility:r,buffer:{type:e,hasDynamicOffset:t,minBindingSize:n}}}CreateSamplerBindingLayout(e,t,n){return t??=this.#i(),{binding:n,visibility:t,sampler:{type:e}}}CreateTextureBindingLayout(e,t,n,r,i){return r??=this.#i(),{binding:i,visibility:r,texture:{sampleType:e,viewDimension:t,multisampled:n}}}CreateStorageTextureBindingLayout(e,t,n,r,i){return r??=this.#i(),{binding:i,visibility:r,storageTexture:{access:t,format:e,viewDimension:n}}}CreateExternalTextureBindingLayout(e,t){return e??=this.#i(),{binding:t,visibility:e,externalTexture:{}}}CreateCommandEncoder(){return this.#n=this.Device.createCommandEncoder({label:this.#t})}GetCommandEncoder(e=!1){if(!this.#n){if(e){const t=`${this.#t&&`Label: "${this.#t}".`}`;mt(te.COMMAND_ENCODER_NOT_FOUND,` ${t} Creating a new one.`)}return this.CreateCommandEncoder()}return this.#n}SubmitCommandBuffer(){this.Device.queue.submit([this.#n.finish()])}CreateBuffer(e){const t=e.label??this.CreateStageLabel("Buffer");return this.Device.createBuffer({label:t,...e})}WriteBuffer(e,t,n=0,r,i){this.Device.queue.writeBuffer(e,n,t,r,i)}CopyBufferToBuffer(e,t,n=t.size,r=0,i=0){this.GetCommandEncoder(!0).copyBufferToBuffer(e,r,t,i,n)}RemovePipeline(e){const t=this.Pipelines.indexOf(e);t<0?(mt(te.PIPELINE_NOT_FOUND,`${this.#e}Pipeline. The following pipeline was not found when
                calling \`${this.#e==="Render"&&`${this.#e}er`||"Computation"}.RemovePipeline\` method.`),console.warn(e)):(this.Pipelines[t].Destroy(),this.Pipelines.splice(t,1))}set CommandEncoder(e){this.#n=e}set CommandEncoderLabel(e){this.#t=e}get Name(){return this.#s}Destroy(){this.Pipelines.forEach(e=>e.Destroy()),this.CommandEncoder=void 0,this.Pipelines.splice(0)}}class rm{#n;Active;#t;Device;#e=[];Reflect;GPUPipeline;constructor(e,t,n){this.#n=n,this.#t=t,this.Active=!0,this.Device=e}CreatePipelineLabel(e){return this.#n&&e&&`${this.#n} ${e}`||""}CreatePipelineLayout(e,t){return t??=this.CreatePipelineLabel(`${this.#t} Pipeline Layout`),e=ut(e),this.Device.createPipelineLayout({label:t,bindGroupLayouts:e})}CreateShaderModule(e,t,n,r){e||(e=Yl,mt(te.SHADER_CODE_NOT_FOUND)),t??=this.CreatePipelineLabel("Shader Module");const i=Array.isArray(e)&&e.join(`

`)||e;return this.Reflect=new Zp(i),this.Device.createShaderModule({label:t,code:i,sourceMap:n,compilationHints:r})}#s(e,t,n=0,r=[]){const{format:i}=e.type,o=e.type.members??i?.members;let a=n+(e.offset??0);if(!o){const l=am((i??e.type).name),u=e.size/lm(l);return new(um(l))(t,a,u)}for(let l=0,u={},c=i?.isStruct&&e.count||1;l<c;++l)o.forEach(h=>u[h.name]=this.#s(h,t,a)),i?.isStruct&&(a+=e.stride),r.push(u);return r.length===1&&r[0]||r}CreateBuffer(e){const t=e.label??this.CreatePipelineLabel("Buffer");return this.Device.createBuffer({label:t,...e})}CreateReadableBuffer(e){const t=typeof e=="number",n=gt.READABLE|(!t&&e.usage||0);let r=t&&e;r||=e.size;const i=e?.label??"Readable Buffer";return this.CreateBuffer({label:i,size:r,...e,usage:n})}CreateWritableBuffer(e){const t=typeof e=="number",n=gt.WRITABLE|(!t&&e.usage||0);let r=t&&e;r||=e.size;const i=e?.label??"Writable Buffer";return this.CreateBuffer({label:i,size:r,...e,usage:n})}CreateUniformBuffer(e,t){!this.Reflect&&ie(te.SHADER_MODULE_NOT_FOUND,`\`${this.#t}Pipeline.CreateUniformBuffer\`.
            Use \`${this.#t}Pipeline.CreateShaderModule\` before creating a uniform buffer.`);const n=this.Reflect.uniforms.find(({name:a})=>e===a);!n&&ie(te.UNIFORM_NOT_FOUND,`\`${e}\` in shader uniforms.`),e==="resolution"&&mt(te.INVALID_UNIFORM_NAME,`\`${e}\`.`);const r=t?.label??`${e} Uniform Buffer`,i=new ArrayBuffer(n.size),o=gt.UNIFORM|t?.usage;return{buffer:this.CreateBuffer({label:r,size:i.byteLength,...t,usage:o}),[e]:this.#s(n,i)}}CreateStorageBuffer(e,t=1){!this.Reflect&&ie(te.SHADER_MODULE_NOT_FOUND,`\`${this.#t}Pipeline.CreateStorageBuffer\`.
            Use \`${this.#t}Pipeline.CreateShaderModule\` before creating a storage buffer.`);const n=this.Reflect.storage.find(({name:d})=>e===d);!n&&ie(te.STORAGE_NOT_FOUND,`\`${e}\` in shader bindings.`);const r=typeof t=="number",i=r&&t||t.length,o=gt.STORAGE|(!r&&t.usage||0),a=!r&&t.label||`${e} Storage Buffer`,l=n.format.size*i,u=new ArrayBuffer(l),c=d=>(Object.keys(d).forEach(w=>{if(d[w].buffer instanceof ArrayBuffer){const I=d[w].constructor,E=l/I.BYTES_PER_ELEMENT;d[w]=new I(u,0,E)}else c(d[w])}),d),h=this.#s(n,u);return{buffer:this.CreateBuffer({label:a,size:l,...t,usage:o}),[e]:h.buffer instanceof ArrayBuffer?new h.constructor(u,0,i):c(h)}}WriteBuffer(e,t,n=0,r,i){this.Device.queue.writeBuffer(e,n,t,r,i)}GetBufferMinBindingSize(e){return!this.Reflect&&ie(te.SHADER_MODULE_NOT_FOUND,`\`${this.#t}Pipeline.GetBufferMinBindingSize\`.
            Use \`${this.#t}Pipeline.CreateShaderModule\` before requesting buffer's min binding size.`),this.Reflect.getBindGroups().flat().find(({name:t})=>e===t)?.size??ie(te.BINDING_NOT_FOUND,`\`${e}\` in shader bind groups.`)}CreateBindGroupEntries(e,t=0){return Array.isArray(e)&&e.map((n,r)=>({binding:t?.[r]??r,resource:n}))||[{binding:t,resource:e}]}CreateBindGroupLayout(e,t){return t??=this.CreatePipelineLabel("Bind Group Layout"),e=Array.isArray(e)&&e.map((n,r)=>({...n,binding:n.binding??r}))||[{...e,binding:e.binding??0}],this.Device.createBindGroupLayout({entries:e,label:t})}CreateBindGroup(e,t=0,n){return n??=this.CreatePipelineLabel("Bind Group"),typeof t=="number"&&(t=this.GPUPipeline?this.GPUPipeline.getBindGroupLayout(t):ie(te.PIPELINE_NOT_FOUND,`${this.#t}Pipeline.
                    Use \`${this.#t}Stage.AddPipeline\` before creating a bind group.`)),this.Device.createBindGroup({entries:e,label:n,layout:t})}SetBindGroups(e,t){const n=Array.isArray(e),r=Array.isArray(t);t=(t=n&&r?t.map(i=>ut(i)):r&&t||t&&[t])&&t||[],this.#e=n&&e.map((i,o)=>({bindGroup:i,dynamicOffsets:t,active:!0}))||[{bindGroup:e,dynamicOffsets:t,active:!0}]}AddBindGroups(e,t){const n=Array.isArray(e),r=Array.isArray(t);return t=(t=n&&r?t.map(i=>ut(i)):r&&t||t&&[t])&&t||[],this.#e.push(...n&&e.map(i=>({bindGroup:i,dynamicOffsets:t,active:!0}))||[{bindGroup:e,dynamicOffsets:t,active:!0}])}SetActiveBindGroups(e){e=ut(e);for(let t=this.#e.length;t--;)this.#e[t].active=e.includes(t)}UseBindGroups(e){for(let t=0,n=0,r=this.#e.length;t<r;++t){const{bindGroup:i,dynamicOffsets:o,active:a}=this.#e[t];a&&e.setBindGroup(n++,i,o)}}GetBindGroupsInfo(){!this.Reflect&&ie(te.SHADER_MODULE_NOT_FOUND,`\`${this.#t}Pipeline.GetBindGroupsInfo\`.
            Use \`${this.#t}Pipeline.CreateShaderModule\` before requesting bind groups information.`);const e=this.#e.length,t=new Array(e),n=this.Reflect.getBindGroups();for(let r=0;r<e;++r){const{bindGroup:{label:i},dynamicOffsets:o,active:a}=this.#e[r];t[r]={label:i,active:a,dynamicOffsets:o,bindings:n[r]}}return t}ClearBindGroups(){this.#e.splice(0)}Destroy(){this.ClearBindGroups()}}class pT extends rm{DestroyPassEncoder=!1;#n=[];#t;#e;TextureView;#s=[0,0,0,0];#i=[0,void 0,void 0,void 0,void 0];constructor(e,t,n){super(e,"Render",n),this.#e=t}async Init(e={}){let t=No(e),{vertex:n,fragment:r}=e;!t&&!n&&(t=this.CreateShaderModule()),t&&(n??=this.CreateVertexState(t),r??=this.CreateFragmentState(t));const i=e.label??this.CreatePipelineLabel("Render Pipeline"),o=e.layout??"auto";return this.GPUPipeline=await this.Device.createRenderPipelineAsync({label:i,layout:o,vertex:n,fragment:r,...e})}CreateBlendComponent(e="add",t="one",n="zero"){return{operation:e,srcFactor:t,dstFactor:n}}CreateColorTargetState(e=this.#e,t,n){return t&&={color:t.color??{},alpha:t.alpha??{}},{format:e,blend:t,writeMask:n}}CreateMultisampleState(e=4,t,n){return{count:e,mask:t,alphaToCoverageEnabled:n}}CreateStencilFaceState(e,t,n,r){return{compare:e,failOp:t,depthFailOp:n,passOp:r}}CreateDepthStencilState(e="depth24plus",t=!0,n="less",r,i,o,a,l,u,c){return{format:e,depthWriteEnabled:t,depthCompare:n,stencilFront:r,stencilBack:i,stencilReadMask:o,stencilWriteMask:a,depthBias:l,depthBiasSlopeScale:u,depthBiasClamp:c}}CreateVertexState(e,t="vertex",n,r){return{module:e,entryPoint:t,buffers:n=ut(n),constants:r}}CreateFragmentState(e,t="fragment",n,r){return n??=this.CreateColorTargetState(),{module:e,entryPoint:t,targets:n=ut(n),constants:r}}#r(e,t=0,n=0){return{format:e,shaderLocation:t,offset:n}}CreateVertexBufferLayout(e,t,n="vertex"){!this.Reflect&&ie(te.SHADER_MODULE_NOT_FOUND,"`RenderPipeline.CreateVertexBufferLayout`.\n            Call `RenderPipeline.CreateShaderModule` before creating a vertex layout or vertex buffer.");const{entry:{vertex:r}}=this.Reflect,i=r.find(({name:l})=>n===l);!i&&ie(te.VERTEX_ENTRY_NOT_FOUND,`\`${n}\` in vertex shader entries.`);let o=[],a=0;for(let l=0,u=(e=ut(e)).length;l<u;++l){const c=e[l],h=typeof c=="string",d=h?c:c.name,w=i.inputs.find(({name:I})=>d===I);if(w){const I=h?im(w.type.size):c.format;o.push(this.#r(I,+w.location,a)),a+=om(I);continue}mt(te.VERTEX_ATTRIBUTE_NOT_FOUND,`\`${d}\` in vertex shader inputs.`)}return{arrayStride:a,stepMode:t,attributes:o}}CreateVertexBuffer(e,t=1,n,r="vertex"){const i=typeof t=="number",o=!i&&t.label||"Vertex Buffer",a=gt.VERTEX|(!i&&t.usage||0);if(e instanceof Float32Array)return this.CreateBuffer({label:o,size:e.byteLength,...t,usage:a});const l=this.CreateVertexBufferLayout(e,n,r),u=(i&&t||(t.count??1))*l.arrayStride;return{buffer:this.CreateBuffer({label:o,size:u,...t,usage:a}),layout:l}}SetVertexBuffers(e,t,n){n=ut(n),t=ut(t),this.#n=Array.isArray(e)&&e.map((r,i)=>({buffer:r,offset:t[i],size:n[i]}))||[{buffer:e,offset:t[0],size:n[0]}]}AddVertexBuffers(e,t,n){return n=ut(n),t=ut(t),this.#n.push(...Array.isArray(e)&&e.map((r,i)=>({buffer:r,offset:t[i],size:n[i]}))||[{buffer:e,offset:t[0],size:n[0]}])}CreateIndexBuffer(e,t){const n=gt.INDEX|t?.usage,r=t?.label??"Index Buffer";return e=Array.isArray(e)&&new Uint32Array(e)||e,this.CreateBuffer({label:r,size:e.byteLength,...t,usage:n})}SetIndexBuffer(e,t="uint32",n,r){this.#t=e&&{buffer:e,format:t,offset:n,size:r}}UseRenderBuffers(e){for(let t=0,n=this.#n.length;t<n;++t){const{buffer:r,offset:i,size:o}=this.#n[t];e.setVertexBuffer(t,r,i,o)}this.#t&&e.setIndexBuffer(this.#t.buffer,this.#t.format,this.#t.offset,this.#t.size)}SetDrawParams(e,t,n,r,i){this.#i[0]=e,this.#i[1]=t,this.#i[2]=n,this.#i[3]=r,this.#i[4]=i,i!==void 0&&(this.#i[3]=i,this.#i[4]=r)}get ColorAttachment(){return 0}set BlendConstant(e){this.#s=cm(e)}get BlendConstant(){return this.#s}get VertexBuffers(){return this.#n}get IndexBuffer(){return this.#t}get DrawMethod(){return this.#t?"drawIndexed":"draw"}get DrawParams(){return this.#i}Destroy(){super.Destroy(),this.DestroyPassEncoder=!1,this.#s=[0,0,0,0],this.#n.forEach(({buffer:e})=>e.destroy()),this.#t?.buffer.destroy(),this.#n.splice(0)}}class mT extends rm{constructor(e,t){super(e,"Compute",t)}async Init(e){const t=e.label??this.CreatePipelineLabel("Compute Pipeline"),n=e.layout??"auto",r=No(e)??this.CreateShaderModule();return this.GPUPipeline=await this.Device.createComputePipelineAsync({label:t,layout:n,compute:{module:r,...e}})}}class gT extends sm{#n=[1];#t;constructor(e,t){super(e,"Compute",t),this.CreatePassDescriptor()}CreatePassDescriptor(e,t){return e??=this.CreateStageLabel("Compute Pass"),this.#t={label:e,timestampWrites:t}}GetMaxEvenWorkgroupDimension(e=1){const{maxComputeInvocationsPerWorkgroup:t}=this.Device.limits;return 0|(e===3?Math.cbrt(t):e===2?Math.sqrt(t):t)}async CreatePipeline(e){const t=Array.isArray(e)||typeof e=="string",n=t||"shader"in e,r=new this.Pipeline(e.pipelineName);return e=No(e)??(n&&r.CreateShaderModule(...Object.values(t&&[e]||e))||e),await this.AddPipeline(r,e),r}async AddPipeline(e,t){return await e.Init(t),Reflect.deleteProperty(e,"Init"),this.Pipelines.push(e),e}#e(e,t){e.setPipeline(t.GPUPipeline),t.UseBindGroups(e),e.dispatchWorkgroups(...this.#n),e.end()}Compute(e=!0){const t=this.GetCommandEncoder().beginComputePass(this.#t),n=this.Pipelines.length;if(n-1||!this.Pipelines[0].Active)for(let r=0;r<n;++r){const i=this.Pipelines[r];i.Active&&this.#e(t,i)}else this.#e(t,this.Pipelines[0]);e&&this.Submit()}Submit(){this.SubmitCommandBuffer(),this.CommandEncoder=void 0}set Workgroups(e){this.#n=ut(e).map(Math.ceil)}get Pipeline(){const{Name:e,Device:t}=this;return class extends mT{constructor(n=e){super(t,n)}}}Destroy(){super.Destroy(),this.Workgroups=1}}class yT extends sm{#n=new Float32Array(3);#t=!1;#e;#s;#i;#r;#a=void 0;#u;#o;#f;#h;#l;constructor(e,t,n,r){super(e,"Render",t),!n&&ie(te.CANVAS_NOT_FOUND);const i=n.getContext("webgpu");!i&&ie(te.CONTEXT_NOT_FOUND),i.configure({device:e,...r}),this.#i=this.CreateBuffer({size:this.#n.length*Float32Array.BYTES_PER_ELEMENT,label:"Render Pipeline Resolution Buffer",usage:gt.UNIFORM}),this.#e=n,this.#s=i,this.#g(),this.#f=r.format,this.CreatePassDescriptor(this.CreateColorAttachment())}ConfigureContext(e){const t=e.format??this.#f;this.#s.configure({device:this.Device,format:t,...e})}CreateColorAttachment(e,t,n="clear",r="store",i,o){return{view:t,loadOp:n,storeOp:r,clearValue:e&&cm(e),resolveTarget:i,depthSlice:o}}CreateStencilAttachment(e=0,t="clear",n="store",r){return{stencilClearValue:e,stencilLoadOp:t,stencilStoreOp:n,stencilReadOnly:r}}CreateDepthStencilAttachment(e,t=1,n="clear",r="store",i,o){return this.#t=!0,this.#r=new Zl(this.Device),{view:e,depthClearValue:t,depthLoadOp:n,depthStoreOp:r,depthReadOnly:i,...o}}#c(){const e=this.CurrentTexture,{width:t,height:n}=e;this.#o&&this.#o.width===t&&this.#o.height===n||(this.#o?.destroy(),this.#o=this.#r.CreateTextureFromSource(e,{sampleCount:this.#h?.sampleCount??1,usage:GPUTextureUsage.RENDER_ATTACHMENT,label:"Depth Texture",format:"depth24plus",mipmaps:!1})),this.#u.depthStencilAttachment.view=this.#o.createView()}CreatePassDescriptor(e,t,n,r,i,o){const a=ut(e);return n??=this.CreateStageLabel("Render Pass"),this.#u={colorAttachments:a,depthStencilAttachment:t,occlusionQuerySet:r,timestampWrites:i,maxDrawCount:o,label:n}}CreateStorageTextureBindingLayout(e=this.#f,t,n,r,i){return{binding:i,visibility:GPUShaderStage.FRAGMENT,storageTexture:{access:t,format:e,viewDimension:n}}}#d(e,t=0,n=0){return{format:e,shaderLocation:t,offset:n}}#g(){this.#n.set([this.#e.width,this.#e.height,this.DevicePixelRatio]),this.WriteBuffer(this.#i,this.#n)}SetCanvasSize(e,t,n=!0){!this.Device&&ie(te.DEVICE_NOT_FOUND),!this.#e&&ie(te.CANVAS_NOT_FOUND);let r=this.DevicePixelRatio*e|0,i=this.DevicePixelRatio*t|0;const o=this.Device.limits.maxTextureDimension2D;r=Math.max(1,Math.min(r,o)),i=Math.max(1,Math.min(i,o)),this.#e.width===r&&this.#e.height===i||(this.#e.height=i,this.#e.width=r,this.#g(),n&&(this.#e.style.width=`${e}px`,this.#e.style.height=`${t}px`))}async CreatePipeline(e,t){const n=Array.isArray(e)||typeof e=="string",r=n||"shader"in e,i=new this.Pipeline(e.pipelineName);return e=No(e)??(r&&i.CreateShaderModule(...Object.values(n&&[e]||e))||e),await this.AddPipeline(i,e,t),i}async AddPipeline(e,t,n){const r=await e.Init(t);return n&&(this.#l?this.#l.setPipeline(r):mt(te.RENDER_PASS_NOT_FOUND)),Reflect.deleteProperty(e,"Init"),this.Pipelines.push(e),e}#p(e,t,n){const r=n||!this.#l;if(!this.#l){let i="view";const o=this.#u.colorAttachments[e.ColorAttachment];this.#h&&(i="resolveTarget",o.view=this.#h.createView()),o[i]=e.TextureView||this.CurrentTextureView,this.#l=this.GetCommandEncoder().beginRenderPass(this.#u)}r&&this.#l.setPipeline(e.GPUPipeline),e.UseRenderBuffers(this.#l),e.UseBindGroups(this.#l),this.#l.setBlendConstant(e.BlendConstant),this.#l[e.DrawMethod](...e.DrawParams),e.DestroyPassEncoder&&!t&&this.DestroyRenderPass()}Render(e=!0){this.#t&&this.#c();const t=this.Pipelines.length;if(t-1||!this.Pipelines[0].Active)for(let n=0;n<t;++n){const r=this.Pipelines[n];r.Active&&this.#p(r,e,!0)}else this.#p(this.Pipelines[0],e,!1);e&&this.Submit()}DestroyRenderPass(){this.#l?.end(),this.#l=void 0}Submit(){this.DestroyRenderPass(),this.SubmitCommandBuffer(),this.CommandEncoder=void 0}get Canvas(){return this.#e}get Context(){return this.#s}get RenderPass(){return this.#l}get DepthTexture(){return this.#o}get CurrentTexture(){return this.#s.getCurrentTexture()}get CurrentTextureView(){return this.CurrentTexture.createView()}set MultisampleTexture(e){this.#h=e}get MultisampleTexture(){return this.#h}set DevicePixelRatio(e){this.#a=e}get DevicePixelRatio(){return this.#a??globalThis.devicePixelRatio??1}get ResolutionBuffer(){return this.#i}get BaseCanvasSize(){const{width:e,height:t}=this.#e,n=this.DevicePixelRatio;return[e/n,t/n]}get CanvasSize(){return[this.#e.width,this.#e.height]}get AspectRatio(){return!this.#e&&ie(te.CANVAS_NOT_FOUND),this.#e.width/this.#e.height}get Pipeline(){const{Name:e,Device:t}=this,n=this.#f;return class extends pT{constructor(r=e){super(t,n,r)}}}Destroy(){super.Destroy(),this.DestroyRenderPass(),this.#i.destroy(),this.#o=this.#o?.destroy(),this.#r=this.#r?.Destroy(),this.#s.unconfigure()}}class zt{static#n=[];static#t=null;static#e=null;static#s={powerPreference:void 0,forceFallbackAdapter:!1};static OnLost;static#i={label:void 0,requiredFeatures:new Set,requiredLimits:void 0};static#r(e){this.#i.label??=e&&`${e} Device`||""}static async CreateQuerySet(e,t){const n=(await this.GPUDevice).createQuerySet({type:e,count:t});return this.#n.push(n),n}static RenderPipeline(e,t="",n={}){return n.format??=this.PreferredCanvasFormat,this.#r(t),(async()=>{const r=await this.GPUDevice;return new Proxy(fT,{construct(i){return new i(r,t,e,n)}})})()}static Renderer(e,t="",n={}){return n.format??=this.PreferredCanvasFormat,this.#r(t),(async()=>{const r=await this.GPUDevice;return new Proxy(yT,{construct(i){return new i(r,t,e,n)}})})()}static ComputePipeline(e=""){return this.#r(e),(async()=>{const t=await this.GPUDevice;return new Proxy(dT,{construct(n){return new n(t,e)}})})()}static Computation(e=""){return this.#r(e),(async()=>{const t=await this.GPUDevice;return new Proxy(gT,{construct(n){return new n(t,e)}})})()}static LegacyTexture(e){return(async()=>{const t=await this.GPUDevice;return new Proxy(Qc,{construct(n){return new Qc(t,e)}})})()}static Texture(e){return(async()=>{const t=await this.GPUDevice;return new Proxy(Zl,{construct(n){return new n(t,e)}})})()}static Destroy(e,t){this.#n.forEach(n=>n.destroy()),(e=ut(e)).forEach(n=>n?.destroy()),(t=ut(t)).forEach(n=>n?.destroy()),this.#t?.destroy(),this.#n.splice(0),this.#i.requiredFeatures.clear(),this.#e=this.#t=null,this.DescriptorLabel=this.RequiredLimits=void 0,this.PowerPreference=this.ForceFallbackAdapter=void 0}static#a(e){if(zt.OnLost)return zt.OnLost(e);const t=(e.message&&` | Message: ${e.message}`)??".";ie(te.DEVICE_LOST,`Reason: ${e.reason}`+t)}static#u(){return!navigator.gpu&&ie(te.WEBGPU_NOT_SUPPORTED),async()=>{const e=await navigator.gpu.requestAdapter(this.#s);return!e&&ie(te.ADAPTER_NOT_FOUND),this.#e=e}}static#o(){return async()=>{const{requiredFeatures:e,requiredLimits:t,label:n}=this.#i,r=await(await this.Adapter).requestDevice({requiredFeatures:e,requiredLimits:t,defaultQueue:{label:n}});return!r&&ie(te.DEVICE_NOT_FOUND),r.lost.then(this.#a),this.#t=r}}static set PowerPreference(e){this.#s.powerPreference=e}static set ForceFallbackAdapter(e){this.#s.forceFallbackAdapter=e}static set DescriptorLabel(e){this.#i.label=e}static async SetRequiredFeatures(e){const t=(await this.Adapter).features;return(e=ut(e)).forEach(n=>t.has(n)?this.#i.requiredFeatures.add(n):mt(te.FEATURE_NOT_FOUND,`"${n}".
It will be skipped when requesting a GPUDevice.`)),this.#i.requiredFeatures}static set RequiredLimits(e){this.#i.requiredLimits=e}static get PreferredCanvasFormat(){return!navigator.gpu&&ie(te.WEBGPU_NOT_SUPPORTED),navigator.gpu.getPreferredCanvasFormat()}static get Adapter(){return(async()=>this.#e??await this.#u()())()}static get GPUDevice(){return(async()=>this.#t??await this.#o()())()}static set OnDeviceLost(e){this.OnLost=e}static get OnDeviceLost(){return this.OnLost}static get Device(){return this.GPUDevice}static get VERSION(){return"0.1.1"}}function Pe(s){for(let e in s)s[e]={value:s[e]};return Object.freeze(Object.create(null,s))}function im(s){switch(s){case 2:return"unorm8x2";case 4:return"float32";case 8:return"float32x2";case 12:return"float32x3";case 16:return"float32x4"}}function om(s){switch(s){case"uint8x2":case"sint8x2":case"unorm8x2":case"snorm8x2":return 2;case"uint32":case"sint32":case"float32":case"uint8x4":case"sint8x4":case"unorm8x4":case"snorm8x4":case"uint16x2":case"sint16x2":case"unorm16x2":case"snorm16x2":case"float16x2":return 4;case"uint16x4":case"sint16x4":case"uint32x2":case"sint32x2":case"unorm16x4":case"snorm16x4":case"float16x4":case"float32x2":return 8;case"uint32x3":case"sint32x3":case"float32x3":return 12;case"uint32x4":case"sint32x4":case"float32x4":return 16}return 0}function am(s){return s==="f16"||s.includes("h")?"f16":s.includes("f")?"f32":s.includes("u")?"u32":"i32"}function lm(s){return+s.slice(1)/8}function um(s){return s==="f16"&&ie(te.FORMAT_NOT_SUPPORTED,`${s}.`),s==="f32"?Float32Array:s==="u32"?Uint32Array:Int32Array}function ut(s){return Array.isArray(s)&&s||[s]}function cm(s){return s instanceof Co?s.rgba:s}function No(s){return s instanceof GPUShaderModule&&s||s.module}Pe({TRIANGLE:3,SQUARE:4,PENTAGON:5,HEXAGON:6,HEPTAGON:7,OCTAGON:8,NONAGON:9,DECAGON:10,DODECAGON:12});console.info("%cUWAL v0.1.1","background:#005a9c;padding:3px;color:#fff;");class bT{unet;sampsCount=0;image;Renderer;seedBuffer;color3f;color4u;totSamps=500;Computation;canvas;denoiserBuffer;denoiserFreq=25;storageBufferSize;workgroupDimension;resizeTimeout;seed;draw=this.render.bind(this);quartSamps=this.totSamps/4;context;constructor(){zt.OnLost=()=>{},Sk(Tk).then(e=>this.unet=e)}resize(e,t){clearTimeout(this.resizeTimeout),this.resizeTimeout=setTimeout(()=>{zt.Destroy([this.color3f.buffer,this.color4u.buffer]),this.create(this.Renderer.Canvas,e,t),this.setOutputCanvas(this.canvas,e,t),this.sampsCount=0},500)}setOutputCanvas(e,t,n){this.canvas=e,this.context=e.getContext("2d"),this.canvas.width=t,this.canvas.height=n,this.image=new ImageData(new Uint8ClampedArray(t*n*4),t,n)}async create(e,t,n){const r=Uint32Array.BYTES_PER_ELEMENT*4;return this.storageBufferSize=t*n*r,await this.checkRequiredLimits(e),this.Renderer=new(await zt.Renderer(e)),this.Renderer.SetCanvasSize(t,n,!1),await this.createComputePipeline(),await this.createRenderPipeline(),requestAnimationFrame(this.draw),[t,n]}async checkRequiredLimits(e){const t=this.storageBufferSize*Uint32Array.BYTES_PER_ELEMENT*4;zt.RequiredLimits={maxStorageBufferBindingSize:t},zt.SetRequiredFeatures("bgra8unorm-storage");try{this.Computation=new(await zt.Computation()),this.workgroupDimension=this.Computation.GetMaxEvenWorkgroupDimension(2)}catch(n){this.create(e,832,624),console.warn(n),console.warn(["Will be used a fallback with the minimum `maxStorageBufferBindingSize`","value available in all WebGPU contexts (134217728 bytes [128 MB]),","which produces a 832 x 624 pixel image."].join(" "))}}async createComputePipeline(){const e=this.createSpheres(),[t,n]=this.Renderer.CanvasSize,r=new this.Computation.Pipeline;await this.Computation.AddPipeline(r,{module:r.CreateShaderModule(`
                const SPHERES = ${e.length}u;
                ${Ik}
            `),constants:{DIMENSION_SIZE:this.workgroupDimension,SAMPLES:4/this.totSamps}});const{seed:i,buffer:o}=r.CreateUniformBuffer("seed");this.color3f=r.CreateStorageBuffer("color3f",this.storageBufferSize*.75),this.denoiserBuffer=r.CreateReadableBuffer(this.storageBufferSize),this.color4u=r.CreateStorageBuffer("color4u",{length:this.storageBufferSize,usage:GPUBufferUsage.COPY_SRC});const{spheres:a,buffer:l}=r.CreateStorageBuffer("spheres",e.length);for(let u=0,c=0;u<e.length;u++,c=u*12)a[u].p.set(e[u].p,c+0),a[u].rad.set(e[u].rad,c+3),a[u].e.set(e[u].e,c+4),a[u].refl.set(e[u].refl,c+7),a[u].c.set(e[u].c,c+8);this.Computation.WriteBuffer(l,a[0].p.buffer),this.seedBuffer=o,this.seed=i,r.SetBindGroups(r.CreateBindGroup(r.CreateBindGroupEntries([this.Renderer.ResolutionBuffer,this.seedBuffer,this.color3f.buffer,this.color4u.buffer,l]))),this.Computation.Workgroups=[t/this.workgroupDimension,n/this.workgroupDimension]}async createRenderPipeline(){const e=new this.Renderer.Pipeline;await this.Renderer.AddPipeline(e,e.CreateShaderModule([Yc.Resolution,Yc.Quad,kk])),e.SetBindGroups(e.CreateBindGroup(e.CreateBindGroupEntries([this.Renderer.ResolutionBuffer,this.color4u.buffer]))),e.SetDrawParams(6)}async render(){this.updateSeedAndSampsBuffer(),this.Computation.Compute(!1),this.Computation.CopyBufferToBuffer(this.color4u.buffer,this.denoiserBuffer),this.Computation.Submit(),this.Renderer.Render(),++this.sampsCount%this.denoiserFreq||await this.denoise(),this.sampsCount<this.quartSamps&&requestAnimationFrame(this.draw)}updateSeedAndSampsBuffer(){this.seed[0]=Math.random()*4294967295,this.seed[1]=Math.random()*4294967295,this.seed[2]=Math.random()*4294967295,this.Computation.WriteBuffer(this.seedBuffer,this.seed)}async denoise(){await this.denoiserBuffer.mapAsync(GPUMapMode.READ),this.image.data.set(new Uint32Array(this.denoiserBuffer.getMappedRange())),this.denoiserBuffer.unmap();const e=this.context;this.unet.tileExecute({done:t=>e.putImageData(t,0,0),color:this.image})}createSpheres(){return[{p:[998,40.8,81.6],rad:[1e3],e:[0,0,0],refl:[0],c:[.8,.2,.2]},{p:[-898,40.8,81.6],rad:[1e3],e:[0,0,0],refl:[0],c:[.2,.2,.8]},{p:[50,40.8,1e3],rad:[1e3],e:[0,0,0],refl:[0],c:[.2,.8,.2]},{p:[50,40.8,-830],rad:[1e3],e:[0,0,0],refl:[0],c:[0,0,0]},{p:[50,1e3,81.6],rad:[1e3],e:[0,0,0],refl:[0],c:[.8,.8,.8]},{p:[50,-1e3+81.6+4.2,81.6],rad:[1e3],e:[0,0,0],refl:[0],c:[.8,.8,.8]},{p:[27,16.5,47],rad:[16.5],e:[0,0,0],refl:[1],c:[.999,.999,.999]},{p:[73,16.5,78],rad:[16.5],e:[0,0,0],refl:[2],c:[.999,.999,.999]},{p:[50,68.16-.27+74.2,81.6],rad:[60],e:[12,12,12],refl:[0],c:[0,0,0]}]}}const sa=new bT;self.onmessage=async({data:s})=>{const{width:e,height:t}=s;switch(s.action){case"Transfer::WebGPU":const[n,r]=await sa.create(s.canvas,e,t);(e!==n||t!==r)&&self.postMessage({width:n,height:r});break;case"Transfer::2D":return sa.setOutputCanvas(s.canvas,e,t);case"Resize::Window":return sa.resize(e,t)}};self.onerror=console.error;
