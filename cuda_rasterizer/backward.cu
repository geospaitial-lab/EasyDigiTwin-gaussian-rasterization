/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include "helper_math.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos,
 const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs,
  glm::vec3* dL_dcampos)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);

	// Gradients of loss w.r.t. campos
	dL_dcampos[idx] = glm::vec3(-dL_dmean.x, -dL_dmean.y, -dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const float kernel_size,
	const float* view_matrix,
	const float* dL_dconics,
	const float2* dL_dqs,
    const float3* dL_dnormals,
	float3* dL_dmeans,
	float* dL_dcov,
	float* dL_dviewmatrix,
	float* dL_dfocal_x,
	float* dL_dfocal_y,
	const float4* __restrict__ conic_opacity,
	float* dL_dopacity)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant
	// intermediate forward results needed in the backward.
	const float3 mean = means[idx];
	const float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	const float2 dL_dq = dL_dqs[idx];
    const float3 dL_dnormal = dL_dnormals[idx];
	const float4 conic = conic_opacity[idx];
	const float combined_opacity = conic.w;
	float3 t = transformPoint4x3(mean, view_matrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	const float l_inv = 1 / length(t);

	glm::mat3 J = glm::mat3(
	    focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		t.x * l_inv, t.y * l_inv, t.z * l_inv);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	const float det_0 = max(1e-6, cov2D[0][0] * cov2D[1][1] - cov2D[0][1] * cov2D[0][1]);
	const float det_1 = max(1e-6, (cov2D[0][0] + kernel_size) * (cov2D[1][1] + kernel_size) - cov2D[0][1] * cov2D[0][1]);
	// sqrt here
	const float coef = sqrt(det_0 / (det_1+1e-6) + 1e-6);

	// update the gradient of alpha and save the gradient of dalpha_dcoef
	// we need opacity as input
	// new_opacity = coef * opacity
	// if we know the new opacity, we can derive original opacity and then dalpha_dcoef = dopacity * opacity
	const float opacity = combined_opacity / (coef + 1e-6);
	const float dL_dcoef = dL_dopacity[idx] * opacity;
	const float dL_dsqrtcoef = dL_dcoef * 0.5 * 1. / (coef + 1e-6);
	const float dL_ddet0 = dL_dsqrtcoef / (det_1+1e-6);
	const float dL_ddet1 = dL_dsqrtcoef * det_0 * (-1.f / (det_1 * det_1 + 1e-6));
	//TODO gradient is zero if det_0 or det_1 < 0
	const float dL_dcov2D_00_coef = dL_ddet0 * cov2D[1][1] + dL_ddet1 * (cov2D[1][1] + kernel_size);
	const float dL_dcov2D_01_coef = dL_ddet0 * (-1. * cov2D[0][1]) + dL_ddet1 * (-1. * cov2D[0][1]);  //also dcoef_dcov2D_10_coef
	const float dL_dcov2D_11_coef = dL_ddet0 * cov2D[0][0] + dL_ddet1 * (cov2D[0][0] + kernel_size);

    const float cov_inv_22_inv = 1 / (det_0 + 1e-6);
    float2 q;
    glm::vec3 ray_space_normal;
    glm::vec3 n;
	if (det_0 <= 1e-6){
	    q = make_float2(0, 0);
	    ray_space_normal = glm::vec3(0, 0, 0);
	    n = glm::vec3(0, 0, 0);
	}
	else{
        q = {(cov2D[1][0] * cov2D[2][1] - cov2D[1][1] * cov2D[2][0]) * cov_inv_22_inv,
             (cov2D[0][1] * cov2D[2][0] - cov2D[0][0] * cov2D[2][1]) * cov_inv_22_inv};
        ray_space_normal = glm::vec3(-q.x, -q.y, -1);
        n = J * ray_space_normal;
	}

	cov2D[0][0] += kernel_size;
	cov2D[1][1] += kernel_size;

	float det2inv = 1 / (det_1 * det_1 + 1e-6);

	if (glm::determinant(cov2D) != 0)
	{
	    glm::mat3 dL_dcov2D = glm::mat3(0.0f);
	    // Gradient w.r.t normalized normals -> gradient w.r.t. unnormalized normals

        const float normal_len_inv = 1 / (length(n) + 1e-6);
        const float normal_len_inv3 = normal_len_inv * normal_len_inv * normal_len_inv;
        const float nux2 = n.x * n.x;
        const float nuy2 = n.y * n.y;
        const float nuz2 = n.z * n.z;
        const float nuxy = n.x * n.y;
        const float nuyz = n.y * n.z;
        const float nuzx = n.z * n.x;

	    float3 dL_du_normal = {0.0f, 0.0f, 0.0f};
        dL_du_normal.x = ((nuz2 + nuy2) * dL_dnormal.x - nuxy * dL_dnormal.y - nuzx * dL_dnormal.z) * normal_len_inv3;
        dL_du_normal.y = ((nuz2 + nux2) * dL_dnormal.y - nuyz * dL_dnormal.z - nuxy * dL_dnormal.x) * normal_len_inv3;
        dL_du_normal.z = ((nuy2 + nux2) * dL_dnormal.z - nuzx * dL_dnormal.x - nuyz * dL_dnormal.y) * normal_len_inv3;



	    // Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
	    dL_dcov2D[0][0] = det2inv * (- cov2D[1][1] * cov2D[1][1] * dL_dconic.x
	                                 + 2 * cov2D[0][1] * cov2D[1][1] * dL_dconic.y
                                     - cov2D[0][1] * cov2D[1][0] * dL_dconic.z);
        dL_dcov2D[1][0] = det2inv * (  cov2D[0][1] * cov2D[1][1] * dL_dconic.x
                                     - (cov2D[0][1] * cov2D[0][1] + cov2D[0][0] * cov2D[1][1]) * dL_dconic.y
                                     + cov2D[0][1] * cov2D[0][0] * dL_dconic.z);
//        dL_dcov2D[0][1] = det2inv * (  cov2D[1][0] * cov2D[1][1] * dL_dconic.x
//                                     - (cov2D[1][0] * cov2D[1][0] + cov2D[0][0] * cov2D[1][1]) * dL_dconic.y
//                                     + cov2D[1][0] * cov2D[0][0] * dL_dconic.z);
        dL_dcov2D[0][1] = dL_dcov2D[1][0];
        dL_dcov2D[1][1] = det2inv * (- cov2D[0][0] * cov2D[0][0] * dL_dconic.z
	                                 + 2 * cov2D[0][1] * cov2D[0][0] * dL_dconic.y
                                     - cov2D[0][1] * cov2D[1][0] * dL_dconic.x);

	    if (det_0 <= 1e-6 || det_1 <= 1e-6){
			dL_dopacity[idx] = 0;
		} else {
			// Gradiends of alpha respect to cov due to low pass filter
			dL_dcov2D[0][0] += dL_dcov2D_00_coef;
			dL_dcov2D[1][0] += dL_dcov2D_01_coef;
			dL_dcov2D[0][1] += dL_dcov2D_01_coef;
			dL_dcov2D[1][1] += dL_dcov2D_11_coef;

			// update dL_dopacity
			dL_dopacity[idx] = dL_dopacity[idx] * coef;
		}

		// q also recieves gradients from normal
		const float dL_dqx = dL_dq.x - (J[0][0] * dL_du_normal.x + J[0][1] * dL_du_normal.y + J[0][2] * dL_du_normal.z);
        const float dL_dqy = dL_dq.y - (J[1][0] * dL_du_normal.x + J[1][1] * dL_du_normal.y + J[1][2] * dL_du_normal.z);

        cov2D[0][0] -= kernel_size;
        cov2D[1][1] -= kernel_size;

		// Gradients of loss w.r.t. entries of 2D covariance matrix
		// coming from q
		dL_dcov2D[0][0] += (cov2D[0][1] * dL_dqy - cov2D[1][1] * dL_dqx) * q.x * cov_inv_22_inv;
		dL_dcov2D[1][0] += (cov2D[0][1] * dL_dqy - cov2D[1][1] * dL_dqx) * q.y * cov_inv_22_inv;
		dL_dcov2D[2][0] += (cov2D[0][1] * dL_dqy - cov2D[1][1] * dL_dqx) * cov_inv_22_inv;
		dL_dcov2D[0][1] += (cov2D[1][0] * dL_dqx - cov2D[0][0] * dL_dqy) * q.x * cov_inv_22_inv;
        dL_dcov2D[1][1] += (cov2D[1][0] * dL_dqx - cov2D[0][0] * dL_dqy) * q.y * cov_inv_22_inv;
        dL_dcov2D[2][1] += (cov2D[1][0] * dL_dqx - cov2D[0][0] * dL_dqy) * cov_inv_22_inv;

        glm::mat3 dL_dVrk = T * glm::transpose(dL_dcov2D) * glm::transpose(T);

        // Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = dL_dVrk[0][0];
		dL_dcov[6 * idx + 3] = dL_dVrk[1][1];
		dL_dcov[6 * idx + 5] = dL_dVrk[2][2];

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry,
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = dL_dVrk[0][1] + dL_dVrk[1][0];
		dL_dcov[6 * idx + 2] = dL_dVrk[0][2] + dL_dVrk[2][0];
		dL_dcov[6 * idx + 4] = dL_dVrk[1][2] + dL_dVrk[2][1];

        glm::mat3 dL_dT = Vrk * T * dL_dcov2D + glm::transpose(Vrk) * T * glm::transpose(dL_dcov2D);
        glm::mat3 dL_dJ = glm::transpose(W) * dL_dT;
        // gradient from normal
        dL_dJ[0][0] += ray_space_normal.x * dL_du_normal.x;
        dL_dJ[0][1] += ray_space_normal.x * dL_du_normal.y;
        dL_dJ[0][2] += ray_space_normal.x * dL_du_normal.z;
        dL_dJ[1][0] += ray_space_normal.y * dL_du_normal.x;
        dL_dJ[1][1] += ray_space_normal.y * dL_du_normal.y;
        dL_dJ[1][2] += ray_space_normal.y * dL_du_normal.z;
        dL_dJ[2][0] += ray_space_normal.z * dL_du_normal.x;
        dL_dJ[2][1] += ray_space_normal.z * dL_du_normal.y;
        dL_dJ[2][2] += ray_space_normal.z * dL_du_normal.z;

        glm::mat3 dL_dW = dL_dT * glm::transpose(J);

        const float tz_inv = 1.f / t.z;
        const float tz_inv2 = tz_inv * tz_inv;
        const float tz_inv3 = tz_inv2 * tz_inv;
        const float tx2 = t.x * t.x;
        const float ty2 = t.y * t.y;
        const float tz2 = t.z * t.z;
        const float l_inv3 = l_inv * l_inv * l_inv;

        // Gradients of loss w.r.t. transformed Gaussian mean t
        const float dL_dtx = x_grad_mul * ((tz2 + ty2) * l_inv3 * dL_dJ[2][0]
                                           - t.y * t.x * l_inv3 * dL_dJ[2][1]
                                           - t.z * t.x * l_inv3 * dL_dJ[2][2]
                                           - focal_x * tz_inv2 * dL_dJ[0][2]);
        const float dL_dty = y_grad_mul * (- t.x * t.y * l_inv3 * dL_dJ[2][0]
                                           + (tz2 + tx2) * l_inv3 * dL_dJ[2][1]
                                           - t.z * t.y * l_inv3 * dL_dJ[2][2]
                                           - focal_y * tz_inv2 * dL_dJ[1][2]);
        const float dL_dtz = - focal_x * tz_inv2 * dL_dJ[0][0]
                             + 2 * focal_x * t.x * tz_inv3 * dL_dJ[0][2]
                             - focal_y * tz_inv2 * dL_dJ[1][1]
                             + 2 * focal_y * t.y * tz_inv3 * dL_dJ[1][2]
                             - t.x * t.z * l_inv3 * dL_dJ[2][0]
                             - t.y * t.z * l_inv3 * dL_dJ[2][1]
                             + (tx2 + ty2) * l_inv3 * dL_dJ[2][2];

        // Gradients of loss w.r.t. focal lengths
        dL_dfocal_x[idx] = x_grad_mul * (dL_dJ[0][0] * tz_inv - dL_dJ[0][2] * t.x * tz_inv2);
        dL_dfocal_y[idx] = y_grad_mul * (dL_dJ[1][1] * tz_inv - dL_dJ[1][2] * t.y * tz_inv2);

        // Gradients of loss w.r.t. view matrix
        dL_dviewmatrix[16 * idx + 0] = dL_dW[0][0] + dL_dtx * mean.x;
        dL_dviewmatrix[16 * idx + 1] = dL_dW[1][0] + dL_dty * mean.x;
        dL_dviewmatrix[16 * idx + 2] = dL_dW[2][0] + dL_dtz * mean.x;
        dL_dviewmatrix[16 * idx + 4] = dL_dW[0][1] + dL_dtx * mean.y;
        dL_dviewmatrix[16 * idx + 5] = dL_dW[1][1] + dL_dty * mean.y;
        dL_dviewmatrix[16 * idx + 6] = dL_dW[2][1] + dL_dtz * mean.y;
        dL_dviewmatrix[16 * idx + 8] = dL_dW[0][2] + dL_dtx * mean.z;
        dL_dviewmatrix[16 * idx + 9] = dL_dW[1][2] + dL_dty * mean.z;
        dL_dviewmatrix[16 * idx + 10] = dL_dW[2][2] + dL_dtz * mean.z;
        dL_dviewmatrix[16 * idx + 12] = dL_dtx;
        dL_dviewmatrix[16 * idx + 13] = dL_dty;
        dL_dviewmatrix[16 * idx + 14] = dL_dtz;

        // Account for transformation of mean to t
        // t = transformPoint4x3(mean, view_matrix);
        float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

        // Gradients of loss w.r.t. Gaussian means, but only the portion
        // that is caused because the mean affects the covariance matrix.
        // Additional mean gradient is accumulated in BACKWARD::preprocess.
        dL_dmeans[idx] = dL_dmean;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
		for (int i = 0; i < 16; i++)
            dL_dviewmatrix[16 * idx + i] = 0;
        dL_dmeans[idx] = make_float3(0);
        dL_dfocal_x[idx] = 0;
        dL_dfocal_y[idx] = 0;
	}
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* view,
	const float* proj,
	const glm::vec3* campos,
	const float4* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_ddepth,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dprojmatrix,
	glm::vec3* dL_dcampos)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// Compute loss gradient w.r.t. projection matrix
	const float mul3 = -(mul1 * dL_dmean2D[idx].x + mul2 * dL_dmean2D[idx].y);
	dL_dprojmatrix[16 * idx + 0] = m.x * m_w * dL_dmean2D[idx].x;
	dL_dprojmatrix[16 * idx + 1] = m.x * m_w * dL_dmean2D[idx].y;
	dL_dprojmatrix[16 * idx + 3] = m.x * mul3;
	dL_dprojmatrix[16 * idx + 4] = m.y * m_w * dL_dmean2D[idx].x;
    dL_dprojmatrix[16 * idx + 5] = m.y * m_w * dL_dmean2D[idx].y;
    dL_dprojmatrix[16 * idx + 7] = m.y * mul3;
    dL_dprojmatrix[16 * idx + 8] = m.z * m_w * dL_dmean2D[idx].x;
    dL_dprojmatrix[16 * idx + 9] = m.z * m_w * dL_dmean2D[idx].y;
    dL_dprojmatrix[16 * idx + 11] = m.z * mul3;
    dL_dprojmatrix[16 * idx + 12] = m_w * dL_dmean2D[idx].x;
    dL_dprojmatrix[16 * idx + 13] = m_w * dL_dmean2D[idx].y;
    dL_dprojmatrix[16 * idx + 15] = mul3;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;
	// Compute gradient updates due to dL_dmeans from depths
	const glm::vec3 depth_vec = ((glm::vec3*)means)[idx] - *campos;
	const glm::vec3 dL_mean_depth = depth_vec / glm::length(depth_vec) * dL_ddepth[idx];

	// That's the third part of the mean gradient.
	dL_dmeans[idx] += dL_mean_depth;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor,
		                  (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh, (glm::vec3*)dL_dcampos);

    // Compute gradient updates due to dL_dcampos from depths
    dL_dcampos[idx] += -dL_mean_depth;

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ subpixel_offset,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float2* __restrict__ qs,
	const float3* __restrict__ normals,
	const float3* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
    const float* __restrict__ dL_dout_depths,
    const float3* __restrict__ dL_dout_normals,
    const float* __restrict__ dL_dout_opacitys,
    const float* __restrict__ dL_dout_distortions,
	float4* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_ddepths,
	float2* __restrict__ dL_dqs,
    float3* __restrict__ dL_dnormals)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	if (inside){
		pixf.x += subpixel_offset[pix_id].x;
		pixf.y += subpixel_offset[pix_id].y;
		// if (pix_id == 0){
		// 	printf("\n\n in backward rendering, pixf is %.5f %.5f  offset %.5f %.5f\n\n", pixf.x, pixf.y, subpixel_offset[pix_id].x, subpixel_offset[pix_id].y);
		// }
	}

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];
	__shared__ float2 collected_qs[BLOCK_SIZE];
    __shared__ float3 collected_normals[BLOCK_SIZE];

	__shared__ float shared_dL_dcolors[C * BLOCK_SIZE];
    __shared__ float shared_dL_ddepths[BLOCK_SIZE];
    __shared__ float2 shared_dL_dqs[BLOCK_SIZE];
    __shared__ float3 shared_dL_dnormals[BLOCK_SIZE];
    __shared__ float shared_dL_dopacity[BLOCK_SIZE];
    __shared__ float4 shared_dL_dmean2D[BLOCK_SIZE];
    __shared__ float3 shared_dL_dconic2D[BLOCK_SIZE];

    __shared__ int skip_counter;

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors.
	const float3 T_D_final = final_Ts[pix_id];
	const float T_final = inside ? T_D_final.x : 0;
	const float A_final = inside ? 1 - T_final : 0;
	const float D_1_final = inside ? T_D_final.y : 0;
	const float D_2_final = inside ? T_D_final.z : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = {0};
	float accum_rest_depth = 0.0f;
	float3 accum_rest_normal = make_float3(0.0f, 0.0f, 0.0f);
	float accum_rest_opacity = 0.0f;
	float accum_rest_dL_dw = 0.0f;
	float dL_dpixel[C];
	float bg_color_local[C];
	float dL_dout_depth;
	float3 dL_dout_normal;
    float dL_dout_opacity;
    float dL_dout_distortion;
	if (inside)
	{
		for (int i = 0; i < C; i++)
		{
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
			bg_color_local[i] = bg_color[i];
		}
			
        dL_dout_depth = dL_dout_depths[pix_id];
        dL_dout_normal = dL_dout_normals[pix_id];
        dL_dout_opacity = dL_dout_opacitys[pix_id];
        dL_dout_distortion = dL_dout_distortions[pix_id];
    }

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_depth = 0;
	float3 last_normal = make_float3(0, 0, 0);

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];

			collected_depths[block.thread_rank()] = depths[coll_id];
			collected_qs[block.thread_rank()] = qs[coll_id];
            collected_normals[block.thread_rank()] = normals[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
		{
		    block.sync();
		    if (block.thread_rank() == 0)
                skip_counter = 0;
            block.sync();

            bool skip = done;

			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;

			skip |= contributor >= last_contributor;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;

		    skip |= power > 0.0f;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);

			skip |= alpha < 1.0f / 255.0f;

			const float pix_depth = collected_depths[j] + (d.x * collected_qs[j].x + d.y * collected_qs[j].y);
			skip |= pix_depth <= near_cutoff;

			if (skip)
			{
			    atomicAdd(&skip_counter, 1);
			}
			block.sync();
			if (skip_counter == BLOCK_SIZE)
			{
			    continue;
			}

			T = skip ? T : T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;  // also dout_depth_ddepth

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = skip ? accum_rec[ch] : last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = skip ? last_color[ch] : c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				shared_dL_dcolors[block.thread_rank() * C + ch] = skip ? 0.0f : dchannel_dcolor * dL_dchannel;
			}

			accum_rest_depth = skip ? accum_rest_depth : last_depth * last_alpha + (1.f - last_alpha) * accum_rest_depth;
            last_depth = skip ? last_depth : pix_depth;
            dL_dalpha += (pix_depth - accum_rest_depth) * dL_dout_depth;

            const float m = far_cutoff / (far_cutoff - near_cutoff) * (1 - near_cutoff / pix_depth);
            const float dm_dpix_depth = far_cutoff * near_cutoff / ((far_cutoff - near_cutoff) * pix_depth * pix_depth);

            float dL_dpix_depth = 2.0f * dchannel_dcolor * (m * A_final - D_1_final) * dm_dpix_depth * dL_dout_distortion;

            const float dL_dw_from_distortion = (m * m * A_final + D_2_final - 2 * m * D_1_final) * dL_dout_distortion;

            dL_dalpha += dL_dw_from_distortion - accum_rest_dL_dw;
            accum_rest_dL_dw = skip ? accum_rest_dL_dw : dL_dw_from_distortion * alpha + (1 - alpha) * accum_rest_dL_dw;

            dL_dpix_depth += dL_dout_depth * dchannel_dcolor;
            shared_dL_ddepths[block.thread_rank()] = skip ? 0.0f : dL_dpix_depth;
            shared_dL_dqs[block.thread_rank()] = skip ? make_float2(0.0f, 0.0f) : dL_dpix_depth * make_float2(d.x, d.y);

            const float3 normal = collected_normals[j];
            accum_rest_normal = skip ? accum_rest_normal : last_normal * last_alpha + (1.f - last_alpha) * accum_rest_normal;
            last_normal = skip ? last_normal : normal;
            const float3 dL_dalpha_from_normal = (normal - accum_rest_normal) * dL_dout_normal;
            dL_dalpha += dL_dalpha_from_normal.x;
            dL_dalpha += dL_dalpha_from_normal.y;
            dL_dalpha += dL_dalpha_from_normal.z;

            shared_dL_dnormals[block.thread_rank()] = skip ? make_float3(0.0f, 0.0f, 0.0f) : dL_dout_normal * dchannel_dcolor;

            accum_rest_opacity = skip ? accum_rest_opacity : last_alpha + (1 - last_alpha) * accum_rest_opacity;
            dL_dalpha += (1 - accum_rest_opacity) * dL_dout_opacity;
			
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = skip ? last_alpha : alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color_local[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update shared gradients w.r.t. 2D mean position of the Gaussian
			shared_dL_dmean2D[block.thread_rank()].x = skip ? 0.0f : (dL_dG * dG_ddelx
			                                                          + dL_dpix_depth * collected_qs[j].x) * ddelx_dx;
            shared_dL_dmean2D[block.thread_rank()].y = skip ? 0.0f : (dL_dG * dG_ddely
                                                                      + dL_dpix_depth * collected_qs[j].y) * ddely_dy;
            shared_dL_dmean2D[block.thread_rank()].z = skip ? 0.0f : abs((dL_dG * dG_ddelx
                                                                      + dL_dpix_depth * collected_qs[j].x) * ddelx_dx);
            shared_dL_dmean2D[block.thread_rank()].w = skip ? 0.0f : abs((dL_dG * dG_ddely
                                                                      + dL_dpix_depth * collected_qs[j].y) * ddely_dy);

			// Update shared gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			shared_dL_dconic2D[block.thread_rank()].x = skip ? 0.0f : -0.5f * gdx * d.x * dL_dG;
            shared_dL_dconic2D[block.thread_rank()].y = skip ? 0.0f : -0.5f * gdx * d.y * dL_dG;
            shared_dL_dconic2D[block.thread_rank()].z = skip ? 0.0f : -0.5f * gdy * d.y * dL_dG;

			// Update shared gradients w.r.t. opacity of the Gaussian
            shared_dL_dopacity[block.thread_rank()] = skip ? 0.0f : G * dL_dalpha;

            // Sum up shared gradients
            block.sync();

            for (int i = block.size() / 2; i > 0; i /= 2)
            {
                if (block.thread_rank() < i)
                {
                    for (int ch = 0; ch < C; ch++)
                    {
                        shared_dL_dcolors[block.thread_rank() * C + ch] += shared_dL_dcolors[(block.thread_rank() + i) * C + ch];
                    }
                    shared_dL_ddepths[block.thread_rank()] += shared_dL_ddepths[block.thread_rank() + i];
                    shared_dL_dmean2D[block.thread_rank()] += shared_dL_dmean2D[block.thread_rank() + i];
                    shared_dL_dconic2D[block.thread_rank()] += shared_dL_dconic2D[block.thread_rank() + i];
                    shared_dL_dopacity[block.thread_rank()] += shared_dL_dopacity[block.thread_rank() + i];
                    shared_dL_dqs[block.thread_rank()] += shared_dL_dqs[block.thread_rank() + i];
                    shared_dL_dnormals[block.thread_rank()] += shared_dL_dnormals[block.thread_rank() + i];
                }
                block.sync();
            }

            // first thread adds summed up gradients to global memory
            if (block.thread_rank() == 0)
            {
                for (int ch = 0; ch < C; ch++)
                {
                    atomicAdd(&(dL_dcolors[global_id * C + ch]), shared_dL_dcolors[ch]);
                }
                atomicAdd(&dL_dmean2D[global_id].x, shared_dL_dmean2D[0].x);
                atomicAdd(&dL_dmean2D[global_id].y, shared_dL_dmean2D[0].y);
                atomicAdd(&dL_dmean2D[global_id].z, shared_dL_dmean2D[0].z);
                atomicAdd(&dL_dmean2D[global_id].w, shared_dL_dmean2D[0].w);
                atomicAdd(&dL_dconic2D[global_id].x, shared_dL_dconic2D[0].x);
                atomicAdd(&dL_dconic2D[global_id].y, shared_dL_dconic2D[0].y);
                atomicAdd(&dL_dconic2D[global_id].w, shared_dL_dconic2D[0].z);
                atomicAdd(&dL_dopacity[global_id], shared_dL_dopacity[0]);
                atomicAdd(&dL_ddepths[global_id], shared_dL_ddepths[0]);
                atomicAdd(&dL_dqs[global_id].x, shared_dL_dqs[0].x);
                atomicAdd(&dL_dqs[global_id].y, shared_dL_dqs[0].y);
                atomicAdd(&dL_dnormals[global_id].x, shared_dL_dnormals[0].x);
                atomicAdd(&dL_dnormals[global_id].y, shared_dL_dnormals[0].y);
                atomicAdd(&dL_dnormals[global_id].z, shared_dL_dnormals[0].z);
            }

		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const float kernel_size,
	const glm::vec3* campos,
	const float4* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_ddepth,
	float2* dL_dq,
    float3* dL_dnormal,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot,
	float* dL_dviewmatrix,
    float* dL_dprojmatrix,
    glm::vec3* dL_dcampos,
    float* dL_dfocal_x,
    float* dL_dfocal_y,
	const float4* conic_opacity,
	float* dL_dopacity)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		kernel_size,
		viewmatrix,
		dL_dconic,
		dL_dq,
		dL_dnormal,
		(float3*)dL_dmean3D,
		dL_dcov3D,
		dL_dviewmatrix,
        dL_dfocal_x,
        dL_dfocal_y,
		conic_opacity,
		dL_dopacity);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		campos,
		(float4*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_ddepth,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot,
		dL_dprojmatrix,
        dL_dcampos);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* subpixel_offset,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* depths,
	const float2* qs,
    const float3* normals,
	const float3* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	const float* dL_dout_depths,
	const float3* dL_dout_normals,
    const float* dL_dout_opacitys,
    const float* dL_dout_distortions,
	float4* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
    float* dL_ddepths,
    float2* dL_dqs,
    float3* dL_dnormals)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		subpixel_offset,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		depths,
		qs,
        normals,
		final_Ts,
		n_contrib,
		dL_dpixels,
        dL_dout_depths,
        dL_dout_normals,
        dL_dout_opacitys,
        dL_dout_distortions,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
        dL_ddepths,
        dL_dqs,
        dL_dnormals
		);
}