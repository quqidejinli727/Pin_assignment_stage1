#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <cstring>
#include <cstdint>
#include "flute.h"

#ifdef _WIN32
#  include <fcntl.h>
#  include <io.h>
#endif

// ====================== 基本数据结构 ======================

struct Point {
    double x;
    double y;
};

struct Polygon {
    int id;                     // 模块编号：1,2,3,...
    std::vector<Point> verts;   // 多边形顶点
};

struct Net {
    std::vector<Point> pins;    // net 上的所有引脚
};

// 网格：用 int 值表示不同模块，0 表示空白
struct Grid {
    int rows = 0;
    int cols = 0;
    double xMin = 0.0;
    double yMin = 0.0;
    double cellW = 1.0;
    double cellH = 1.0;
    std::vector<int> data;      // rows * cols, row-major

    int& at(int r, int c) {
        return data[r * cols + c];
    }
    const int& at(int r, int c) const {
        return data[r * cols + c];
    }
};

// ====================== 几何辅助函数 ======================

// 奇偶规则判断点是否在多边形内
bool pointInPolygon(double x, double y, const Polygon& poly) {
    bool inside = false;
    size_t n = poly.verts.size();
    if (n < 3) return false;

    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        double xi = poly.verts[i].x, yi = poly.verts[i].y;
        double xj = poly.verts[j].x, yj = poly.verts[j].y;

        bool intersect = ((yi > y) != (yj > y)) &&
                         (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

// 根据模块和引脚坐标，构建并填充网格：模块格子 = 模块 id，空白格子 = 0

Grid buildGrid(const std::vector<Polygon>& modules,
               const std::vector<Net>& nets,
               double gridSize)
{
    double xMin = 1e30, yMin = 1e30;
    double xMax = -1e30, yMax = -1e30;

    auto updateBBox = [&](double x, double y) {
        xMin = std::min(xMin, x);
        yMin = std::min(yMin, y);
        xMax = std::max(xMax, x);
        yMax = std::max(yMax, y);
    };

    // 模块外包框
    for (const auto& m : modules) {
        for (const auto& p : m.verts) {
            updateBBox(p.x, p.y);
        }
    }
    // 引脚也纳入外包框，保证网格覆盖所有端点
    for (const auto& net : nets) {
        for (const auto& p : net.pins) {
            updateBBox(p.x, p.y);
        }
    }

    Grid g;

    if (xMax < xMin || yMax < yMin) {
        // 没有数据
        return g;
    }

    double dx = (xMax - xMin);
    double dy = (yMax - yMin);
    int cols = std::max(1, static_cast<int>(std::ceil(dx / gridSize)));
    int rows = std::max(1, static_cast<int>(std::ceil(dy / gridSize)));

    g.rows = rows;
    g.cols = cols;
    g.xMin = xMin;
    g.yMin = yMin;
    g.cellW = gridSize;
    g.cellH = gridSize;
    g.data.assign(rows * cols, 0); 

    // 为网格赋值模块 id(存在于模块内)
    for (const auto& m : modules) {
        for (int r = 0; r < rows; ++r) {
            double cy = yMin + (r + 0.5) * gridSize;
            for (int c = 0; c < cols; ++c) {
                double cx = xMin + (c + 0.5) * gridSize;
                if (pointInPolygon(cx, cy, m)) {
                    g.at(r, c) = m.id; 
                }
            }
        }
    }

    return g;
}

// 仅基于模块本身构建/填充网格（不把 nets pins 纳入 bbox）。
// 用途：session 批量多 net 时，模块分布不变，网格可缓存复用。
Grid buildGridModulesOnly(const std::vector<Polygon>& modules, double gridSize) {
    double xMin = 1e30, yMin = 1e30;
    double xMax = -1e30, yMax = -1e30;

    auto updateBBox = [&](double x, double y) {
        xMin = std::min(xMin, x);
        yMin = std::min(yMin, y);
        xMax = std::max(xMax, x);
        yMax = std::max(yMax, y);
    };

    for (const auto& m : modules) {
        for (const auto& p : m.verts) {
            updateBBox(p.x, p.y);
        }
    }

    Grid g;
    if (xMax < xMin || yMax < yMin) {
        return g;
    }

    double dx = (xMax - xMin);
    double dy = (yMax - yMin);
    int cols = std::max(1, static_cast<int>(std::ceil(dx / gridSize)));
    int rows = std::max(1, static_cast<int>(std::ceil(dy / gridSize)));

    g.rows = rows;
    g.cols = cols;
    g.xMin = xMin;
    g.yMin = yMin;
    g.cellW = gridSize;
    g.cellH = gridSize;
    g.data.assign(rows * cols, 0);

    for (const auto& m : modules) {
        for (int r = 0; r < rows; ++r) {
            double cy = yMin + (r + 0.5) * gridSize;
            for (int c = 0; c < cols; ++c) {
                double cx = xMin + (c + 0.5) * gridSize;
                if (pointInPolygon(cx, cy, m)) {
                    g.at(r, c) = m.id;
                }
            }
        }
    }

    return g;
}

// 仅基于模块 bbox 计算实际使用的 gridSize，保证 pad 到 512x512。
// 注意：这里不再把 nets pins 纳入 bbox，以换取网格可缓存复用。
double computeGridSizeFromModulesOnly(const std::vector<Polygon>& modules) {
    double xMin = 1e30, yMin = 1e30;
    double xMax = -1e30, yMax = -1e30;
    auto updateBBox = [&](double x, double y) {
        xMin = std::min(xMin, x);
        yMin = std::min(yMin, y);
        xMax = std::max(xMax, x);
        yMax = std::max(yMax, y);
    };
    for (const auto& m : modules) {
        for (const auto& p : m.verts) updateBBox(p.x, p.y);
    }
    if (xMax < xMin || yMax < yMin) return 1.0;
    double dx = (xMax - xMin);
    double dy = (yMax - yMin);
    double dMax = std::max(dx, dy);
    return (dMax > 0.0) ? (dMax / 512.0) : 1.0;
}

// 将网格零填充到 512x512，保持原有坐标对齐
Grid padGridTo512(const Grid& src) {
    Grid dst;
    if (src.rows <= 0 || src.cols <= 0) {
        // 空网格，直接生成全0的 512x512
        dst.rows = 512;
        dst.cols = 512;
        dst.xMin = src.xMin;
        dst.yMin = src.yMin;
        dst.cellW = src.cellW > 0 ? src.cellW : 1.0;
        dst.cellH = src.cellH > 0 ? src.cellH : 1.0;
        dst.data.assign(dst.rows * dst.cols, 0);
        return dst;
    }

    dst.rows = 512;
    dst.cols = 512;
    dst.xMin = src.xMin;
    dst.yMin = src.yMin;
    dst.cellW = src.cellW;
    dst.cellH = src.cellH;
    dst.data.assign(dst.rows * dst.cols, 0);

    int copyRows = std::min(src.rows, dst.rows);
    int copyCols = std::min(src.cols, dst.cols);
    for (int r = 0; r < copyRows; ++r) {
        for (int c = 0; c < copyCols; ++c) {
            dst.at(r, c) = src.at(r, c);
        }
    }
    return dst;
}

// 世界坐标 -> 网格索引 (r, c)
std::pair<int, int> worldToCell(const Grid& g, double x, double y) {
    int c = static_cast<int>(std::floor((x - g.xMin) / g.cellW));
    int r = static_cast<int>(std::floor((y - g.yMin) / g.cellH));

    if (g.cols <= 0 || g.rows <= 0) return {0, 0};

    if (c < 0) c = 0;
    if (c >= g.cols) c = g.cols - 1;
    if (r < 0) r = 0;
    if (r >= g.rows) r = g.rows - 1;
    return {r, c};
}

// 在网格上沿一条线段扫描，统计“格子数值变化次数”
int walkSegmentCountChanges(const Grid& g,
                            double x1, double y1,
                            double x2, double y2)
{
    if (g.rows == 0 || g.cols == 0) return 0;

    auto rc1 = worldToCell(g, x1, y1);
    auto rc2 = worldToCell(g, x2, y2);
    int r1 = rc1.first, c1 = rc1.second;
    int r2 = rc2.first, c2 = rc2.second;

    if (r1 == r2 && c1 == c2) {
        // 起终同一格，不产生跨格变化
        return 0;
    }
    int changes = 0;
    int r = r1;
    int c = c1;
    int lastVal = g.at(r, c);

    auto stepTo = [&](int rr, int cc) {
        int val = g.at(rr, cc);
        if (val != lastVal) {
            if (lastVal == 0 || val == 0) {
                ++changes;
            } else {
                changes += 2;
            }
            lastVal = val;
        }
    };

    // 判断是水平还是垂直线段
    if (r1 == r2) {
        int step = (c2 > c1) ? 1 : -1;
        for (c = c1 + step; ; c += step) {
            stepTo(r, c);
            if (c == c2) break;
        }
    } else {
        int step = (r2 > r1) ? 1 : -1;
        for (r = r1 + step; ; r += step) {
            stepTo(r, c);
            if (r == r2) break;
        }
    }

    return changes;
}

//  为了避免有些引脚在模块内部或模块边界被赋予模块id导致误判，临时将 net 的引脚所在格子置为 0，并在析构时恢复原值
struct TempClearCells {
    Grid& g;
    std::unordered_map<int,int> saved; // key = r*cols + c -> original id

    TempClearCells(Grid& grid, const Net& net) : g(grid) {
        if (g.cols <= 0 || g.rows <= 0) return;
        for (const auto& p : net.pins) {
            auto rc = worldToCell(g, p.x, p.y);
            int r = rc.first;
            int c = rc.second;
            int idx = r * g.cols + c;
            if (saved.find(idx) == saved.end()) {
                saved[idx] = g.data[idx];
                g.data[idx] = 0;
            }
        }
    }

    ~TempClearCells() {
        for (const auto &kv : saved) {
            int idx = kv.first;
            int val = kv.second;
            // 恢复到 grid.data
            if (idx >= 0 && idx < static_cast<int>(g.data.size())) {
                g.data[idx] = val;
            }
        }
    }
};

// 对单个 net 计算 feedthrough：遍历Steiner树中所有边，在网格上扫描，累积“数值变化次数”,feedthrough = (变化次数 / 2)
int feedthroughForNet(const Grid& g, const Net& net) {
    using namespace Flute;

    int d = static_cast<int>(net.pins.size());
    if (d < 2) return 0;

    std::vector<DTYPE> xs(d), ys(d);
    for (int i = 0; i < d; ++i) {
        xs[i] = static_cast<DTYPE>(std::lround(net.pins[i].x));
        ys[i] = static_cast<DTYPE>(std::lround(net.pins[i].y));
    }

    Tree t = flute(d, xs.data(), ys.data(), FLUTE_ACCURACY);

    int totalChanges = 0;
    int branchCount = 2 * t.deg - 2;

    for (int i = 0; i < branchCount; ++i) {
        int j = t.branch[i].n;
        if (i < j && j >= 0 && j < branchCount) {
            double x1 = static_cast<double>(t.branch[i].x);
            double y1 = static_cast<double>(t.branch[i].y);
            double x2 = static_cast<double>(t.branch[j].x);
            double y2 = static_cast<double>(t.branch[j].y);

            int changes = walkSegmentCountChanges(g, x1, y1, x2, y2);
            totalChanges += changes;
        }
    }

    free_tree(t);
    //向下取整弥补部分精度
    int feedthrough = (totalChanges) / 2;
    return feedthrough;
}

int main(int argc, char** argv) {
    using namespace Flute;

    auto hasFlag = [&](const std::string& f) -> bool {
        for (int i = 1; i < argc; ++i) {
            if (argv[i] && std::string(argv[i]) == f) return true;
        }
        return false;
    };

    bool binMode = hasFlag("--bin");

#ifdef _WIN32
    // 兼容 Windows：二进制协议必须禁用 \r\n 转换
    if (binMode) {
        _setmode(_fileno(stdin), _O_BINARY);
        _setmode(_fileno(stdout), _O_BINARY);
    }
#endif

    auto readExact = [&](char* dst, size_t n) -> bool {
        size_t got = 0;
        while (got < n) {
            std::cin.read(dst + got, static_cast<std::streamsize>(n - got));
            std::streamsize r = std::cin.gcount();
            if (r <= 0) return false;
            got += static_cast<size_t>(r);
        }
        return true;
    };

    auto writeAll = [&](const char* src, size_t n) -> bool {
        std::cout.write(src, static_cast<std::streamsize>(n));
        std::cout.flush();
        return static_cast<bool>(std::cout);
    };

    auto readU32 = [&](uint32_t &v) -> bool {
        uint32_t tmp = 0;
        if (!readExact(reinterpret_cast<char*>(&tmp), sizeof(tmp))) return false;
        v = tmp;
        return true;
    };

    auto writeU32 = [&](uint32_t v) -> bool {
        return writeAll(reinterpret_cast<const char*>(&v), sizeof(v));
    };

    // 通过标准输入传入 modules/nets，并通过标准输出返回结果。
    // 兼容：
    //   - 文本模式：ftpred - - -
    //   - 二进制模式：ftpred - - - --bin
    if (argc < 4 || std::string(argv[1]) != "-" || std::string(argv[2]) != "-" || std::string(argv[3]) != "-") {
        std::cerr << "Invalid args. argc=" << argc << " argv=";
        for (int i = 0; i < argc; ++i) {
            std::cerr << "[" << i << ":" << (argv[i] ? argv[i] : "<null>") << "]";
        }
        std::cerr << std::endl;
        std::cerr << "Usage: " << argv[0] << " - - -" << std::endl;
        std::cerr << "stdin format: <modules> then a line '---NETS---' then <nets>" << std::endl;
        std::cerr << "stdout format: one line per net: 'Net i feedthrough = x'" << std::endl;
        return 2;
    }

    // ====================== 二进制帧协议 ======================
    // 帧格式： [u32 payload_len][payload bytes]
    // payload:
    //   msgType(u8): 1=MODULES, 2=NETS, 3=QUIT
    //   for MODULES:
    //     u32 nModules
    //     repeat nModules:
    //       i32 id
    //       u32 vcount
    //       repeat vcount: f64 x, f64 y
    //   for NETS:
    //     u32 nNets
    //     repeat nNets:
    //       u32 pcount
    //       repeat pcount: f64 x, f64 y
    // response (对应 NETS)：
    //   一帧： [u32 payload_len][payload]
    //   payload: u32 nNets + i32 feedthrough[nNets]
    if (binMode) {
        std::ios::sync_with_stdio(false);
        std::cin.tie(nullptr);

        std::vector<Polygon> modules;
        readLUT();

        bool baseGridReady = false;
        Grid baseGrid;
        double gridSizeModulesOnly = 1.0;

        while (true) {
            uint32_t frameLen = 0;
            if (!readU32(frameLen)) {
                break; // EOF
            }
            std::string payload;
            payload.resize(frameLen);
            if (frameLen > 0 && !readExact(&payload[0], frameLen)) {
                break;
            }
            if (payload.empty()) {
                continue;
            }
            const uint8_t msgType = static_cast<uint8_t>(payload[0]);

            const char* p = payload.data() + 1;
            const char* end = payload.data() + payload.size();

            auto need = [&](size_t n) -> bool { return (p + n) <= end; };
            auto read_i32 = [&](int32_t &v) -> bool {
                if (!need(sizeof(int32_t))) return false;
                std::memcpy(&v, p, sizeof(int32_t));
                p += sizeof(int32_t);
                return true;
            };
            auto read_u32 = [&](uint32_t &v) -> bool {
                if (!need(sizeof(uint32_t))) return false;
                std::memcpy(&v, p, sizeof(uint32_t));
                p += sizeof(uint32_t);
                return true;
            };
            auto read_f64 = [&](double &v) -> bool {
                if (!need(sizeof(double))) return false;
                std::memcpy(&v, p, sizeof(double));
                p += sizeof(double);
                return true;
            };

            if (msgType == 3) {
                break;
            } else if (msgType == 1) {
                // MODULES
                modules.clear();
                uint32_t nModules = 0;
                if (!read_u32(nModules)) {
                    std::cerr << "[ftpred-bin] malformed MODULES" << std::endl;
                    break;
                }
                modules.reserve(nModules);
                for (uint32_t i = 0; i < nModules; ++i) {
                    int32_t id = 0;
                    uint32_t vcount = 0;
                    if (!read_i32(id) || !read_u32(vcount)) {
                        std::cerr << "[ftpred-bin] malformed module header" << std::endl;
                        break;
                    }
                    Polygon poly;
                    poly.id = static_cast<int>(id);
                    poly.verts.reserve(vcount);
                    for (uint32_t vi = 0; vi < vcount; ++vi) {
                        double x=0, y=0;
                        if (!read_f64(x) || !read_f64(y)) {
                            std::cerr << "[ftpred-bin] malformed module verts" << std::endl;
                            break;
                        }
                        poly.verts.push_back({x,y});
                    }
                    modules.push_back(std::move(poly));
                }

                // reset caches
                gridSizeModulesOnly = computeGridSizeFromModulesOnly(modules);
                baseGridReady = false;
                baseGrid = Grid();
                continue;
            } else if (msgType == 2) {
                // NETS block
                std::vector<Net> nets;
                uint32_t nNets = 0;
                if (!read_u32(nNets)) {
                    std::cerr << "[ftpred-bin] malformed NETS" << std::endl;
                    break;
                }
                nets.reserve(nNets);
                for (uint32_t ni = 0; ni < nNets; ++ni) {
                    uint32_t pcount = 0;
                    if (!read_u32(pcount)) {
                        std::cerr << "[ftpred-bin] malformed net header" << std::endl;
                        break;
                    }
                    Net net;
                    net.pins.reserve(pcount);
                    for (uint32_t pi = 0; pi < pcount; ++pi) {
                        double x=0, y=0;
                        if (!read_f64(x) || !read_f64(y)) {
                            std::cerr << "[ftpred-bin] malformed pin" << std::endl;
                            break;
                        }
                        net.pins.push_back({x,y});
                    }
                    nets.push_back(std::move(net));
                }

                // 空 block：直接返回空结果帧
                if (nets.empty()) {
                    uint32_t outLen = sizeof(uint32_t);
                    writeU32(outLen);
                    uint32_t outN = 0;
                    writeAll(reinterpret_cast<const char*>(&outN), sizeof(outN));
                    continue;
                }

                if (!baseGridReady) {
                    baseGrid = padGridTo512(buildGridModulesOnly(modules, gridSizeModulesOnly));
                    baseGridReady = true;
                }

                std::vector<int32_t> fts;
                fts.resize(nets.size());

                Grid g = baseGrid;
                for (size_t i = 0; i < nets.size(); ++i) {
                    TempClearCells guard(g, nets[i]);
                    int ft = feedthroughForNet(g, nets[i]);
                    fts[i] = static_cast<int32_t>(ft);
                }

                uint32_t outLen = sizeof(uint32_t) + static_cast<uint32_t>(fts.size() * sizeof(int32_t));
                writeU32(outLen);
                uint32_t outN = static_cast<uint32_t>(fts.size());
                writeAll(reinterpret_cast<const char*>(&outN), sizeof(outN));
                if (!fts.empty()) {
                    writeAll(reinterpret_cast<const char*>(fts.data()), fts.size() * sizeof(int32_t));
                }
                continue;
            } else {
                std::cerr << "[ftpred-bin] unknown msgType=" << int(msgType) << std::endl;
                break;
            }
        }

        deleteLUT();
        return 0;
    }

    std::vector<Polygon> modules;

    auto parseModulesFromStream = [&](std::istream &in) -> bool {
        int nModules = 0;
        if (!(in >> nModules)) {
            std::cerr << "Failed to read number of modules from stream" << std::endl;
            return false;
        }
        std::string line;
        std::getline(in, line); 
        for (int i = 0; i < nModules; ++i) {
            if (!std::getline(in, line)) break;
            if (line.empty()) { --i; continue; }
            std::istringstream iss(line);
            Polygon poly;
            int id;
            std::string name;
            int vcount;
            if (!(iss >> id >> name >> vcount)) {
                std::cerr << "Malformed module line: " << line << std::endl;
                return false;
            }
            poly.id = id;
            for (int vi = 0; vi < vcount; ++vi) {
                double x,y; if (!(iss >> x >> y)) { std::cerr << "Malformed vertex data" << std::endl; return false; }
                poly.verts.push_back({x,y});
            }
            modules.push_back(std::move(poly));
        }
        return true;
    };
    //解析输入。

    auto parseOneNetsBlockFromStream = [&](std::istream &in, std::vector<Net> &outNets) -> bool {
        outNets.clear();
        int nNets = 0;
        if (!(in >> nNets)) {
            std::cerr << "Failed to read number of nets from stream" << std::endl;
            return false;
        }
        std::string line;
        std::getline(in, line);
        for (int i = 0; i < nNets; ++i) {
            if (!std::getline(in, line)) {
                std::cerr << "Unexpected EOF while reading nets" << std::endl;
                return false;
            }
            if (line.empty()) { --i; continue; }
            std::istringstream iss(line);
            int pcount;
            if (!(iss >> pcount)) { std::cerr << "Malformed net line" << std::endl; return false; }
            Net net;
            for (int pi = 0; pi < pcount; ++pi) {
                double x,y;
                if (!(iss >> x >> y)) { std::cerr << "Malformed pin coords" << std::endl; return false; }
                net.pins.push_back({x,y});
            }
            outNets.push_back(std::move(net));
        }
        return true;
    };

    // ========== streaming 解析：modules 一次；nets 多次 ==========
    // stdin format:
    //   <modules>
    //   ---NETS---
    //   <nets block #1>
    //   ---NETS---
    //   <nets block #2>
    //   ...
    //   ---QUIT---
    // 每个 nets block 内部格式与旧版本相同：第一行 nNets，之后 nNets 行 each: pcount x y ...

    std::ostringstream modsBuf;
    {
        std::string line;
        bool foundSep = false;
        while (std::getline(std::cin, line)) {
            if (line == "---NETS---") {
                foundSep = true;
                break;
            }
            if (line == "---QUIT---") {
                return 0;
            }
            modsBuf << line << "\n";
        }
        if (!foundSep) {
            std::cerr << "Missing ---NETS--- separator" << std::endl;
            return 1;
        }
    }

    {
        std::istringstream modsStream(modsBuf.str());
        if (!parseModulesFromStream(modsStream)) return 1;
    }

    readLUT();

    std::ostream* outp = &std::cout;
    std::string netsAccum;
    std::vector<Net> nets;

    // ========== 性能优化（批量 nets）：缓存模块网格 ==========
    // 关键点：modules 不变时，模块填充网格可以只做一次。
    // 这里把 gridSize 与 bbox 固定为“仅由 modules 决定”，避免每个 nets block 重建网格。
    double gridSizeModulesOnly = computeGridSizeFromModulesOnly(modules);
    bool baseGridReady = false;
    Grid baseGrid;

    auto processOneBlock = [&](const std::string& blockText) -> bool {
        std::istringstream netsStream(blockText);
        if (!parseOneNetsBlockFromStream(netsStream, nets)) return false;

        // 空 block（nNets=0）：直接输出 END，避免 session warmup 被“构建 baseGrid”阻塞。
        // 这样 Python 端握手可以立刻确认协议可用。
        if (nets.empty()) {
            (*outp) << "---END---" << "\n";
            outp->flush();
            return true;
        }

        // Lazy init：第一次遇到非空 nets block 时才构建并缓存模块网格。
        if (!baseGridReady) {
            baseGrid = padGridTo512(buildGridModulesOnly(modules, gridSizeModulesOnly));
            baseGridReady = true;
        }

        // 每个 block 仅拷贝一次 baseGrid（512x512 int），避免重新 pointInPolygon 填充。
        Grid g = baseGrid;

        for (size_t ni = 0; ni < nets.size(); ++ni) {
            const Net &net = nets[ni];
            TempClearCells guard(g, net);
            int ft = feedthroughForNet(g, net);
            (*outp) << "Net " << ni << " feedthrough = " << ft << "\n";
        }
        // 每个 block 结束标记，供常驻会话端判断输出边界
        (*outp) << "---END---" << "\n";
        outp->flush();
        return true;
    };

    // 循环读取后续 nets block
    {
        std::string line;
        bool inBlock = false;
        bool seenAnyNetSepAfterModules = false;
        while (std::getline(std::cin, line)) {
            if (line == "---QUIT---") {
                break;
            }
            if (line == "---NETS---") {
                if (!seenAnyNetSepAfterModules) {
                    // modules 后的第一个 ---NETS---：作为“开始 block”的标记
                    // 注意：session 端可能会先发送一个空 block（0 nets）用于握手。
                    seenAnyNetSepAfterModules = true;
                    inBlock = true;
                    netsAccum.clear();
                    continue;
                }

                // 之后每次遇到 ---NETS---：结束上一块（即使为空也要处理，保证输出 ---END---），并开始下一块
                if (!processOneBlock(netsAccum)) {
                    deleteLUT();
                    return 1;
                }
                netsAccum.clear();
                inBlock = true;
                continue;
            }

            if (!inBlock) {
                // modules 段之后理论上应立即看到 ---NETS---；如果有空行等，忽略。
                continue;
            }

            // 累积 nets block 内容
            netsAccum += line;
            netsAccum += "\n";
        }

        // QUIT/EOF 前如果已经进入过 block：无论 netsAccum 是否为空，都处理一次
        // - 空块：输出 ---END---，保证上层不会卡死
        if (inBlock) {
            if (!processOneBlock(netsAccum)) {
                deleteLUT();
                return 1;
            }
        }
    }

    deleteLUT();
    return 0;
}
