## Set parameters

k = 11  # number of nearest neighbors to use
d = 50  # number of singular values to compute
N = 10  # number of recommendations to give

mean.score = 2.75 # for score normalization


## Load movie names, ratings and target user ratings

movie.data = read.csv('ml-latest-small/movies.csv', header=TRUE)[,c(1,2)]
num.of.movies = max(movie.data$movieId)
rating.data = read.csv('ml-latest-small/ratings.csv', header=TRUE)[,-4]
user.data = read.table('testuser', sep='|', header=FALSE)
num.of.user.ratings = nrow(user.data)


## Load libraries

library('Matrix')  # for sparse matrix support
library('irlba')   # for fast SVD


## Convert the user ratings CSV to an MM file and load it as a sparse matrix

convert.to.mm <- function(data, output_file) {
    no_movies = max(data$movieId)
    no_users = max(data$userId)
    no_entries = nrow(data)

    # normalize
    data[,3] <- data[,3] - mean.score

    fileConn <- file(output_file)
    writeLines(c("%%MatrixMarket matrix coordinate real general",
               paste('', no_users, no_movies, no_entries)), fileConn)
    close(fileConn)

    write.table(data, append=TRUE, file=output_file, sep='\t',
                row.names=rep('', no_entries), col.names=FALSE, quote=FALSE)
}

convert.to.mm(rating.data, 'rating.MM')
ratings.matrix = readMM('rating.MM')


## Compute the rank d SVD of the user ratings matrix...

ratings.matrix.svd = irlba(ratings.matrix, nv=d)


## and use it to compute the reduced ratings matrix and...

u = ratings.matrix.svd$u
v = ratings.matrix.svd$v
root.s = Matrix(0, nrow=d, ncol=d, sparse=TRUE)
diag(root.s) = sqrt(ratings.matrix.svd$d)
reduced.ratings = u %*% root.s


## the reduced user ratings vector.

user.rating.to.vec <- function(user.data, movie.data, SVD) {
    vec = Matrix(0, nrow=1, ncol=nrow(SVD$v))
    for (i in (1:nrow(user.data))) {
        movie.name = as.character(user.data[i, 1])
        movie.rating = user.data[i, 2] - mean.score
        movie.id = movie.data[which(movie.data$title == movie.name),]$movieId
        vec[1, movie.id] = movie.rating
    }
    vec
}

user.vec <- user.rating.to.vec(user.data, movie.data, ratings.matrix.svd)
root.s <- Matrix(0, nrow=nrow(v), ncol=d, sparse=TRUE)
diag(root.s) <- sqrt(ratings.matrix.svd$d)
reduced.user.vec <- sqrt(ratings.matrix.svd$d) * (t(v) %*% t(user.vec))


## Next, we find the users closest to our target user in the reduced matrix
## using cosine similarity, and use those users' highest rated movies to
## generate recommendations for the target.

## TODO: Optimize and clean up.
dist <- reduced.ratings %*% reduced.user.vec
row.norm <- function(x) {
    sqrt(sum(x*x))
}
row.norms <- apply(dist, 1, row.norm)
dist <- dist / row.norms
neighbors <- which(dist >= sort(dist, decreasing=T)[k])

rating.sum.vec = Matrix(0, nrow=1, ncol=num.of.movies)

for (i in (1:length(neighbors))) {
    rating.sum.vec = rating.sum.vec + ratings.matrix[i,]
}
top.n.ids <- which(dist >= sort(rating.sum.vec, decreasing=T)[N+num.of.user.ratings])
already.seen.ids = user.data$movieIds
top.n.ids <- top.n.ids[!(top.n.ids %in% already.seen.ids)][(1:N)]

print(as.character(movie.data[(movie.data$movieId %in% top.n.ids),]$title), sep='\n')
