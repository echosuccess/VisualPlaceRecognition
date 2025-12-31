#!/bin/bash
# 还需要运行的Baseline实验命令
# 已完成：CosPlace + L2 + SF-XS, Tokyo-XS

echo "========================================"
echo "剩余30个实验命令"
echo "========================================"

# ============================================
# 1. CosPlace组 - 剩余6个实验
# ============================================
echo ""
echo "[1/30] CosPlace + L2 + SVOX-sun"
python VPR-methods-evaluation/main.py \
  --method cosplace --backbone ResNet18 --descriptors_dimension 512 \
  --image_size 512 512 \
  --database_folder data/svox/images/test/gallery \
  --queries_folder data/svox/images/test/queries \
  --distance_metric l2 \
  --log_dir baseline/cosplace_l2_svox_sun \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[2/30] CosPlace + L2 + SVOX-night"
python VPR-methods-evaluation/main.py \
  --method cosplace --backbone ResNet18 --descriptors_dimension 512 \
  --image_size 512 512 \
  --database_folder data/svox/images/test/gallery \
  --queries_folder data/svox/images/test/queries_night \
  --distance_metric l2 \
  --log_dir baseline/cosplace_l2_svox_night \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[3/30] CosPlace + Dot product + SF-XS"
python VPR-methods-evaluation/main.py \
  --method cosplace --backbone ResNet18 --descriptors_dimension 512 \
  --image_size 512 512 \
  --database_folder data/sf_xs/test/database \
  --queries_folder data/sf_xs/test/queries \
  --distance_metric dot_product \
  --log_dir baseline/cosplace_dotproduct_sf_xs_test \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[4/30] CosPlace + Dot product + Tokyo-XS"
python VPR-methods-evaluation/main.py \
  --method cosplace --backbone ResNet18 --descriptors_dimension 512 \
  --image_size 512 512 \
  --database_folder data/tokyo_xs/test/database \
  --queries_folder data/tokyo_xs/test/queries \
  --distance_metric dot_product \
  --log_dir baseline/cosplace_dotproduct_tokyo_xs_test \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[5/30] CosPlace + Dot product + SVOX-sun"
python VPR-methods-evaluation/main.py \
  --method cosplace --backbone ResNet18 --descriptors_dimension 512 \
  --image_size 512 512 \
  --database_folder data/svox/images/test/gallery \
  --queries_folder data/svox/images/test/queries \
  --distance_metric dot_product \
  --log_dir baseline/cosplace_dotproduct_svox_sun \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[6/30] CosPlace + Dot product + SVOX-night"
python VPR-methods-evaluation/main.py \
  --method cosplace --backbone ResNet18 --descriptors_dimension 512 \
  --image_size 512 512 \
  --database_folder data/svox/images/test/gallery \
  --queries_folder data/svox/images/test/queries_night \
  --distance_metric dot_product \
  --log_dir baseline/cosplace_dotproduct_svox_night \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

# ============================================
# 2. NetVLAD组 - 8个实验
# ============================================
echo ""
echo "[7/30] NetVLAD + L2 + SF-XS"
python VPR-methods-evaluation/main.py \
  --method netvlad --backbone VGG16 --descriptors_dimension 4096 \
  --image_size 512 512 \
  --database_folder data/sf_xs/test/database \
  --queries_folder data/sf_xs/test/queries \
  --distance_metric l2 \
  --log_dir baseline/netvlad_l2_sf_xs_test \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[8/30] NetVLAD + L2 + Tokyo-XS"
python VPR-methods-evaluation/main.py \
  --method netvlad --backbone VGG16 --descriptors_dimension 4096 \
  --image_size 512 512 \
  --database_folder data/tokyo_xs/test/database \
  --queries_folder data/tokyo_xs/test/queries \
  --distance_metric l2 \
  --log_dir baseline/netvlad_l2_tokyo_xs_test \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[9/30] NetVLAD + L2 + SVOX-sun"
python VPR-methods-evaluation/main.py \
  --method netvlad --backbone VGG16 --descriptors_dimension 4096 \
  --image_size 512 512 \
  --database_folder data/svox/images/test/gallery \
  --queries_folder data/svox/images/test/queries \
  --distance_metric l2 \
  --log_dir baseline/netvlad_l2_svox_sun \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[10/30] NetVLAD + L2 + SVOX-night"
python VPR-methods-evaluation/main.py \
  --method netvlad --backbone VGG16 --descriptors_dimension 4096 \
  --image_size 512 512 \
  --database_folder data/svox/images/test/gallery \
  --queries_folder data/svox/images/test/queries_night \
  --distance_metric l2 \
  --log_dir baseline/netvlad_l2_svox_night \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[11/30] NetVLAD + Dot product + SF-XS"
python VPR-methods-evaluation/main.py \
  --method netvlad --backbone VGG16 --descriptors_dimension 4096 \
  --image_size 512 512 \
  --database_folder data/sf_xs/test/database \
  --queries_folder data/sf_xs/test/queries \
  --distance_metric dot_product \
  --log_dir baseline/netvlad_dotproduct_sf_xs_test \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[12/30] NetVLAD + Dot product + Tokyo-XS"
python VPR-methods-evaluation/main.py \
  --method netvlad --backbone VGG16 --descriptors_dimension 4096 \
  --image_size 512 512 \
  --database_folder data/tokyo_xs/test/database \
  --queries_folder data/tokyo_xs/test/queries \
  --distance_metric dot_product \
  --log_dir baseline/netvlad_dotproduct_tokyo_xs_test \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[13/30] NetVLAD + Dot product + SVOX-sun"
python VPR-methods-evaluation/main.py \
  --method netvlad --backbone VGG16 --descriptors_dimension 4096 \
  --image_size 512 512 \
  --database_folder data/svox/images/test/gallery \
  --queries_folder data/svox/images/test/queries \
  --distance_metric dot_product \
  --log_dir baseline/netvlad_dotproduct_svox_sun \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[14/30] NetVLAD + Dot product + SVOX-night"
python VPR-methods-evaluation/main.py \
  --method netvlad --backbone VGG16 --descriptors_dimension 4096 \
  --image_size 512 512 \
  --database_folder data/svox/images/test/gallery \
  --queries_folder data/svox/images/test/queries_night \
  --distance_metric dot_product \
  --log_dir baseline/netvlad_dotproduct_svox_night \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

# ============================================
# 3. MixVPR组 - 8个实验
# ============================================
echo ""
echo "[15/30] MixVPR + L2 + SF-XS"
python VPR-methods-evaluation/main.py \
  --method mixvpr --backbone ResNet50 --descriptors_dimension 4096 \
  --image_size 320 320 \
  --database_folder data/sf_xs/test/database \
  --queries_folder data/sf_xs/test/queries \
  --distance_metric l2 \
  --log_dir baseline/mixvpr_l2_sf_xs_test \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[16/30] MixVPR + L2 + Tokyo-XS"
python VPR-methods-evaluation/main.py \
  --method mixvpr --backbone ResNet50 --descriptors_dimension 4096 \
  --image_size 320 320 \
  --database_folder data/tokyo_xs/test/database \
  --queries_folder data/tokyo_xs/test/queries \
  --distance_metric l2 \
  --log_dir baseline/mixvpr_l2_tokyo_xs_test \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[17/30] MixVPR + L2 + SVOX-sun"
python VPR-methods-evaluation/main.py \
  --method mixvpr --backbone ResNet50 --descriptors_dimension 4096 \
  --image_size 320 320 \
  --database_folder data/svox/images/test/gallery \
  --queries_folder data/svox/images/test/queries \
  --distance_metric l2 \
  --log_dir baseline/mixvpr_l2_svox_sun \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[18/30] MixVPR + L2 + SVOX-night"
python VPR-methods-evaluation/main.py \
  --method mixvpr --backbone ResNet50 --descriptors_dimension 4096 \
  --image_size 320 320 \
  --database_folder data/svox/images/test/gallery \
  --queries_folder data/svox/images/test/queries_night \
  --distance_metric l2 \
  --log_dir baseline/mixvpr_l2_svox_night \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[19/30] MixVPR + Dot product + SF-XS"
python VPR-methods-evaluation/main.py \
  --method mixvpr --backbone ResNet50 --descriptors_dimension 4096 \
  --image_size 320 320 \
  --database_folder data/sf_xs/test/database \
  --queries_folder data/sf_xs/test/queries \
  --distance_metric dot_product \
  --log_dir baseline/mixvpr_dotproduct_sf_xs_test \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[20/30] MixVPR + Dot product + Tokyo-XS"
python VPR-methods-evaluation/main.py \
  --method mixvpr --backbone ResNet50 --descriptors_dimension 4096 \
  --image_size 320 320 \
  --database_folder data/tokyo_xs/test/database \
  --queries_folder data/tokyo_xs/test/queries \
  --distance_metric dot_product \
  --log_dir baseline/mixvpr_dotproduct_tokyo_xs_test \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[21/30] MixVPR + Dot product + SVOX-sun"
python VPR-methods-evaluation/main.py \
  --method mixvpr --backbone ResNet50 --descriptors_dimension 4096 \
  --image_size 320 320 \
  --database_folder data/svox/images/test/gallery \
  --queries_folder data/svox/images/test/queries \
  --distance_metric dot_product \
  --log_dir baseline/mixvpr_dotproduct_svox_sun \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[22/30] MixVPR + Dot product + SVOX-night"
python VPR-methods-evaluation/main.py \
  --method mixvpr --backbone ResNet50 --descriptors_dimension 4096 \
  --image_size 320 320 \
  --database_folder data/svox/images/test/gallery \
  --queries_folder data/svox/images/test/queries_night \
  --distance_metric dot_product \
  --log_dir baseline/mixvpr_dotproduct_svox_night \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

# ============================================
# 4. MegaLoc组 - 8个实验
# ============================================
echo ""
echo "[23/30] MegaLoc + L2 + SF-XS"
python VPR-methods-evaluation/main.py \
  --method megaloc --backbone Dinov2 --descriptors_dimension 8448 \
  --image_size 224 224 \
  --database_folder data/sf_xs/test/database \
  --queries_folder data/sf_xs/test/queries \
  --distance_metric l2 \
  --log_dir baseline/megaloc_l2_sf_xs_test \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[24/30] MegaLoc + L2 + Tokyo-XS"
python VPR-methods-evaluation/main.py \
  --method megaloc --backbone Dinov2 --descriptors_dimension 8448 \
  --image_size 224 224 \
  --database_folder data/tokyo_xs/test/database \
  --queries_folder data/tokyo_xs/test/queries \
  --distance_metric l2 \
  --log_dir baseline/megaloc_l2_tokyo_xs_test \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[25/30] MegaLoc + L2 + SVOX-sun"
python VPR-methods-evaluation/main.py \
  --method megaloc --backbone Dinov2 --descriptors_dimension 8448 \
  --image_size 224 224 \
  --database_folder data/svox/images/test/gallery \
  --queries_folder data/svox/images/test/queries \
  --distance_metric l2 \
  --log_dir baseline/megaloc_l2_svox_sun \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[26/30] MegaLoc + L2 + SVOX-night"
python VPR-methods-evaluation/main.py \
  --method megaloc --backbone Dinov2 --descriptors_dimension 8448 \
  --image_size 224 224 \
  --database_folder data/svox/images/test/gallery \
  --queries_folder data/svox/images/test/queries_night \
  --distance_metric l2 \
  --log_dir baseline/megaloc_l2_svox_night \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[27/30] MegaLoc + Dot product + SF-XS"
python VPR-methods-evaluation/main.py \
  --method megaloc --backbone Dinov2 --descriptors_dimension 8448 \
  --image_size 224 224 \
  --database_folder data/sf_xs/test/database \
  --queries_folder data/sf_xs/test/queries \
  --distance_metric dot_product \
  --log_dir baseline/megaloc_dotproduct_sf_xs_test \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[28/30] MegaLoc + Dot product + Tokyo-XS"
python VPR-methods-evaluation/main.py \
  --method megaloc --backbone Dinov2 --descriptors_dimension 8448 \
  --image_size 224 224 \
  --database_folder data/tokyo_xs/test/database \
  --queries_folder data/tokyo_xs/test/queries \
  --distance_metric dot_product \
  --log_dir baseline/megaloc_dotproduct_tokyo_xs_test \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[29/30] MegaLoc + Dot product + SVOX-sun"
python VPR-methods-evaluation/main.py \
  --method megaloc --backbone Dinov2 --descriptors_dimension 8448 \
  --image_size 224 224 \
  --database_folder data/svox/images/test/gallery \
  --queries_folder data/svox/images/test/queries \
  --distance_metric dot_product \
  --log_dir baseline/megaloc_dotproduct_svox_sun \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "[30/30] MegaLoc + Dot product + SVOX-night"
python VPR-methods-evaluation/main.py \
  --method megaloc --backbone Dinov2 --descriptors_dimension 8448 \
  --image_size 224 224 \
  --database_folder data/svox/images/test/gallery \
  --queries_folder data/svox/images/test/queries_night \
  --distance_metric dot_product \
  --log_dir baseline/megaloc_dotproduct_svox_night \
  --num_preds_to_save 20 --recall_values 1 5 10 20 \
  --save_for_uncertainty --num_workers 8 --batch_size 32

echo ""
echo "========================================"
echo "所有30个剩余实验命令完成！"
echo "========================================"

