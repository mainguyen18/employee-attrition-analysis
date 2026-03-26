# Hướng dẫn tạo slide thuyết trình

## Cấu trúc

- `presentation.tex`: File chính để build slide
- `preamble_beamer.tex`: Preamble riêng cho Beamer (theme, packages)
- Các file section trong thư mục `slides/`:
  - `01_overview.tex`: Tổng quan nghiên cứu
  - `02_data_method.tex`: Dữ liệu và phương pháp
  - `03_eda.tex`: Phân tích khám phá
  - `04_evaluation.tex`: Đánh giá mô hình
  - `05_odds_ratio.tex`: Odds Ratio và diễn giải
  - `06_conclusion.tex`: Kết luận

## Cách build

### Sử dụng pdflatex (khuyến nghị)

```bash
cd report
pdflatex presentation.tex
pdflatex presentation.tex  # Chạy 2 lần để cập nhật TOC
```

### Sử dụng latexmk (tự động)

```bash
cd report
latexmk -pdf presentation.tex
```

## Tùy chỉnh

### Thay đổi theme

Mở `preamble_beamer.tex` và thay đổi:
```latex
\usetheme{Madrid}  % Có thể đổi thành: Berlin, Darmstadt, default, ...
```

### Thay đổi tỷ lệ màn hình

Trong `presentation.tex`, thay đổi:
```latex
\documentclass[aspectratio=169,12pt]{beamer}  % 169 cho màn hình rộng
% Hoặc
\documentclass[aspectratio=43,12pt]{beamer}   % 43 cho màn hình vuông
```

### Thêm/sửa slide

Chỉnh sửa các file trong `slides/` hoặc thêm file mới và `\input` vào `presentation.tex`.

## Lưu ý

- File PDF sẽ được tạo ra: `presentation.pdf`
- Các file build (.aux, .nav, .snm, ...) sẽ bị ignore bởi `.gitignore`
- Đảm bảo các hình ảnh trong `assets/figures/` đã được generate trước khi build

